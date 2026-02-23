import pandas as pd
import numpy as np
import skmob
from skmob.preprocessing import detection

def stop_detection(df : pd.DataFrame, col_lat : str = 'lat', col_lon : str = 'lon', col_uid : str = 'uid', col_time : str = 'datetime',
                   min_minutes_stop : float = 20.0, stop_spatial_radius_km : float = 0.05) -> pd.DataFrame :
    tdf = skmob.TrajDataFrame(df, latitude=col_lat, longitude=col_lon, user_id=col_uid, datetime=col_time)
    #display(tdf)

    stdf = detection.stay_locations(tdf,
                                    stop_radius_factor=0.5,
                                    minutes_for_a_stop=min_minutes_stop,
                                    spatial_radius_km=stop_spatial_radius_km, 
                                    leaving_time=True)
    stdf['duration_secs'] = stdf['leaving_datetime'] -  stdf['datetime']
    stdf['duration_secs'] = stdf['duration_secs'].dt.total_seconds()
    #display(stdf)

    return stdf


def move_detection(df_traj : pd.DataFrame, df_stops : pd.DataFrame) -> pd.DataFrame:

    DEBUG_ID_USER = 0

    
    # 1 - Sort the data in df_traj according to datetime and uid (required by pandas 'merge_asof').
    df_traj_copy = df_traj.loc[:, ['datetime', 'uid']].copy()
    df_traj_copy['orig_order'] = df_traj_copy.index
    df_traj_copy = df_traj_copy.sort_values(by=['datetime', 'uid'])

    # 2 - Sort the data in df_stops according to the starting instant of a stop and uid (required by pandas 'merge_asof').
    df_stops_copy = (
        df_stops.loc[:, ['uid', 'datetime', 'leaving_datetime']].copy()
                .sort_values(["datetime", "uid"])
                .rename(columns={"datetime": "stop_start"})
    )

    #display(df_traj_copy)
    #display(df_stops_copy)



    # 3 -For each trajectory sample, match the latest 'stop_start <= datetime' within same uid.
    merged = pd.merge_asof(
        df_traj_copy,
        df_stops_copy,
        left_on="datetime",
        right_on="stop_start",
        by="uid",
        direction="backward",
        allow_exact_matches=True,
    )
    # display(merged.loc[merged['uid'] == DEBUG_ID_USER].head(50))
    del df_traj_copy, df_stops, df_stops_copy


    # 4 - Find out if a sample falls within a stop segment: if so, remove it from merged.
    merged = merged.loc[~((merged['datetime'] >= merged['stop_start']) & (merged['datetime'] < merged['leaving_datetime']))]
    # display(merged.loc[merged['uid'] == DEBUG_ID_USER].head(50))

    # 5 - Assing an ID to each move.
    merged['move_id'] = merged.groupby(['uid', 'stop_start']).ngroup()
    # display(merged.loc[merged['uid'] == DEBUG_ID_USER].head(50))

    # 6 - Turn the move_id column into a Series indexed by the original order in df_traj.
    merged = merged.set_index('orig_order')['move_id']
    # display(merged)



    # 7 - Add the move_id column to df_traj, keeping only the samples that belong to a move segment.
    df_traj_copy = df_traj.loc[merged.index] 
    df_traj_copy['move_id'] = merged
    df_traj_copy.sort_index(inplace=True)



    return df_traj_copy