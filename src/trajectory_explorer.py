# Plot how the users move over time.
import folium
from folium.plugins import TimestampedGeoJson

def load_dataset_trajectories(path_dataset : str, user_id : int|None = None):
    #user_id = 2
    #path_dataset = './data_simulator/huge_dataset/dataset_simulator_trajectories.parquet'

    # Read the parquet file containing the simulator's trajectories.
    # NOTE: we are selecting only the rows that satisfy the conditions in 'sel' to save memory!
    sel = [("ID", "in", [user_id])] if user_id is not None else []
    df_trajs = pd.read_parquet(path_dataset, filters = sel)
    df_trajs = gpd.GeoDataFrame(df_trajs, geometry=gpd.points_from_xy(df_trajs.lon, df_trajs.lat), crs="EPSG:4326")

    #display(df_trajs.info())
    #isplay(df_trajs)
    
    return df_trajs


def generate_time_slider_map(df_trajs):
    '''
    Generate a folium map with a time slider to visualize user movements over time.

    Parameters:
    
    df_users: DataFrame containing the trajectories of several users. The expected columns are ['ID', 'timestamp', 'geometry'],
              where 'geometry' is a Point.
    '''

    # 2. Pick a palette of colors and map each unique ID to one
    palette = [
        "red", "blue", "green", "purple", "orange",
        "darkred", "lightred", "beige", "darkblue",
        "darkgreen", "cadetblue", "darkpurple"
    ]
    unique_ids = df_trajs["ID"].unique()
    color_map = {
        uid: palette[i % len(palette)]
        for i, uid in enumerate(unique_ids)
    }


    # 2. Build a GeoJSON-style dict with times
    features = []
    for _, row in df_trajs.iterrows():
        cid = row["ID"]
        features.append({
            "type": "Feature",
            "geometry": row.geometry.__geo_interface__,
            "properties": {
                "time": row.timestamp.isoformat(),
                "icon": "circle",
                "iconstyle": {
                    "fillColor": color_map[cid],
                    "fillOpacity": 0.7,
                    "stroke": False,
                    "radius": 10
                },
                "popup": f"ID {cid}<br>{row.timestamp:%Y-%m-%d %H:%M}"
            }
        })

    geojson = {"type": "FeatureCollection", "features": features}


    # 3. Create the map
    m = folium.Map(location=[df_trajs.geometry.y.mean(), df_trajs.geometry.x.mean()], zoom_start=13, control_scale=True)

    # 4. Add the time‐slider layer
    duration_step_minutes = (df_trajs.iloc[1]['timestamp'] - df_trajs.iloc[0]['timestamp']).seconds // 60
    TimestampedGeoJson(
        data=geojson,
        transition_time=200,      # milliseconds between frames
        period=f"PT{duration_step_minutes}M",            # matches your n-minutes stepping
        duration=f"PT{duration_step_minutes}M",
        add_last_point=False,
        auto_play=False,
        loop=False,
    ).add_to(m)


    # 5. Display or save
    m
    # m.save("trajectory_time_slider.html")