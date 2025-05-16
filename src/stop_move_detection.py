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


def move_detection(df : pd.DataFrame, stdf : pd.DataFrame,
                   col_lat : str = 'lat', col_lon : str = 'lon', col_uid : str = 'uid', col_time : str = 'datetime') :
    tdf = skmob.TrajDataFrame(df, latitude=col_lat, longitude=col_lon, user_id=col_uid, datetime=col_time)
    #display(tdf)
    
    ### move detection ###
    trajs = tdf.copy()
    starts = stdf.copy()
    ends = stdf.copy()
    
    trajs.set_index(['uid','datetime'], inplace = True)
    starts.set_index(['uid','datetime'], inplace = True)
    ends.set_index(['uid','leaving_datetime'], inplace = True)
    
    traj_ids = trajs.index
    start_ids = starts.index
    end_ids = ends.index
    
    # some datetime into stdf are approximated. In order to retrieve moves, we have to check the exact datime into 
    # trajectory dataframe. We use `isin()` method to reduce time computation
    traj_df = pd.DataFrame(traj_ids, columns=['trajs'])
    start_df = pd.DataFrame(start_ids, columns=['start'])
    end_df = pd.DataFrame(end_ids, columns=['end'])
    
    start_df['is_in_traj'] = start_df['start'].isin(traj_df['trajs'])
    end_df['is_in_traj'] = end_df['end'].isin(traj_df['trajs'])
    
    start_df['end'] = end_df['end']
    start_df['is_in_traj_end'] = end_df['is_in_traj']
    
    # remove stops which aren't into tdf
    start_df = start_df[(start_df['is_in_traj']!=False)|(start_df['is_in_traj_end']!=False)]
    
    # save index of incomplete stops and convert them into MultiIndex
    incomplete_end = start_df['end'][(start_df['is_in_traj']==False)&(start_df['is_in_traj_end']==True)] 
    incomplete_start = start_df['start'][(start_df['is_in_traj']==True)&(start_df['is_in_traj_end']==False)]
    
    if not incomplete_end.empty:
        incomplete_end = pd.MultiIndex.from_tuples(incomplete_end)
    
    if not incomplete_start.empty:
        incomplete_start = pd.MultiIndex.from_tuples(incomplete_start)
    
    # save complete index
    start_df = start_df[(start_df['is_in_traj']==True)&(start_df['is_in_traj_end']==True)] 
    
    new_start = pd.MultiIndex.from_tuples(start_df['start'])
    new_end = pd.MultiIndex.from_tuples(start_df['end'])
    new_start.set_names(['uid','datetime'],inplace=True)
    new_end.set_names(['uid','datetime'],inplace=True)
    
    # set start and end of stops (using two columns in order to avoid overlaps)
    trajs['start_stop'] = np.nan
    trajs.loc[new_start, 'start_stop'] = 1
    trajs['end_stop'] = np.nan
    trajs.loc[new_end, 'end_stop'] = 1
    
    trajs.reset_index(inplace=True)
    start_idx = trajs[trajs['start_stop']==1].index.to_list()
    end_idx = trajs[trajs['end_stop']==1].index.to_list()
    
    # set incomplete index
    starts_ = [traj_ids.get_loc(e).start - 1 for e in incomplete_end]
    ends_ = [traj_ids.get_loc(s).start + 1 for s in incomplete_start]
    
    if starts_ != []:
        start_idx = start_idx + starts_
    
    if ends_ != []:
        end_idx = end_idx + ends_
    
    trajs['move_id'] = np.nan
    
    for i, (s, e) in enumerate(zip(start_idx,end_idx), 1):
        trajs.loc[s : e+1, 'move_id'] = i
    
    
    trajs['move_id'] = trajs['move_id'].ffill()
    trajs['move_id'] = trajs['move_id'].fillna(0)
    trajs.loc[(trajs['start_stop']==1) | (trajs['end_stop']==1), 'move_id'] = -1
    moves = trajs.loc[trajs['move_id']!=-1]
    
    # NOTE: the final moves result set will be a pandas DataFrame built from the skmob dataframe.
    moves.drop(columns = ['start_stop', 'end_stop'], inplace = True)
    moves['move_id'] = moves['move_id'].astype(np.uint32)
    move_df = pd.DataFrame(moves)