import pandas as pd
from sqlalchemy import create_engine, MetaData, delete
import numpy as np

import time
import pickle
import os
import sys
from tqdm import tqdm

from datetime import timedelta, datetime
import utils



thresh = 100
interval = 45

location = sys.argv[1]
year = int(sys.argv[2])
needed_session = sys.argv[3]

track_config = pd.read_csv('config/track_config.csv')
driver_data = pd.read_csv(f'config/driver_config_{year}.csv')

track = track_config.query(f''' circuit_location == '{location}' ''')

circuit_length = int(track.circuit_length.iloc[0])
corners = eval(track.corners.iloc[0])
start_line = eval(track.start_line.iloc[0])
before_start_line = eval(track.before_start_line.iloc[0])
after_start_line = eval(track.after_start_line.iloc[0])

#Session and circuit information

session = utils.get_session(location, year)
session_names = session.session_name[:2].tolist() #gave up on getting exactly fp1 & fp2 for knn, just getting the first two sessions of the weekend now
print(session_names)

session_key_fp1 = session.query(f'''session_name == '{session_names[0]}' ''').session_key.iloc[0]
session_key_fp2 = session.query(f'''session_name == '{session_names[1]}' ''').session_key.iloc[0]

session_key_race = session.query(f" session_name == '{needed_session}'").session_key.iloc[0]
race_start_time = pd.to_datetime(session.query(f" session_name == '{needed_session}'").date_start.iloc[0])
race_end_time = pd.to_datetime(session.query(f" session_name == '{needed_session}'").date_end.iloc[0])


driver_data = pd.read_csv(f'config/driver_config_{year}.csv')
driver_config = {}
driver_config['driver_code'] = {}
driver_config['driver_number'] = {}
driver_config['team_name'] = {}
driver_config['team_colour'] = {}
driver_config['team_order'] = {}
driver_config['driver_order'] = {}

for ind, driver in driver_data.sort_values(by = ['team_order', 'driver_number']).iterrows():
    driver_config['driver_number'][driver['name_acronym']] = driver['driver_number']
    driver_config['driver_code'][driver['driver_number']] = driver['name_acronym']
    driver_config['team_name'][driver['name_acronym']]= f'''{driver['team_name']}'''
    driver_config['team_colour'][driver['name_acronym']]= f'''#{driver['team_colour']}'''
    driver_config['team_order'][driver['name_acronym']] = driver['team_order']
    driver_config['driver_order'][driver['name_acronym']] = driver['driver_order']


pkl_filename = f"knn/knn_{location}-{year}_FP1_FP2_top25.pkl"
track_layout_filename = f"track_layout/{location}-{year}.csv"
utils.save_knn_pickle(pkl_filename, track_layout_filename, session_key_fp1, session_key_fp2, start_line, before_start_line, after_start_line, LAP_THRESHOLDS = 25)

with open(pkl_filename, 'rb') as file:
    knn, start_line, before_start_line, after_start_line = pickle.load(file)

starting_grid = utils.get_starting_grid(session_key_race, race_start_time) 

# Connect to your SQL database
db_file = f"data/{session_key_race}.db"
if os.path.isfile(db_file):
    os.remove(db_file)
    print(f"Removing older DB file {db_file}")
print(f"Connecting to db using {db_file}")
engine = create_engine(f"sqlite:///{db_file}")

data = {}
data['data'] = {}
data['lap_number'] = {}
for driver_code, driver_number in driver_config['driver_number'].items():
  data['data'][driver_number] = pd.DataFrame()
  data['lap_number'][driver_code] = [1, 0]

# for timestamp in pd.date_range(race_start_time, race_end_time, freq = f'{interval}s'):
for timestamp in pd.date_range(race_start_time + timedelta(minutes = 1), race_end_time, freq = f'{interval}s'):

    # st = ses_start_time + timedelta(seconds=30)
    
    # while True:
    #   et = st + timedelta(seconds = interval)
    #   print(st,et)
    #   if et>ses_end_time:
    #     break

    t1 = time.time()
    st = timestamp
    et = st + timedelta(seconds = interval)
    
    weather_data = utils.get_weather_data(session_key_race, st, et)
    laptimes_data = utils.get_laptimes_data(session_key_race, st, et)
    position_data = utils.get_position_data(session_key_race, et)
    if len(weather_data):
        weather_data.map(str).to_sql('weather', engine, if_exists = 'append', index = False)
    if len(laptimes_data):
        laptimes_data.map(str).to_sql('laptimes', engine, if_exists = 'append', index = False)
    if len(position_data):
        position_data.map(str).to_sql('position', engine, if_exists = 'replace', index = False)
    
    car_data, location_data = utils.get_data_channels({'start_time' : st, 'end_time' : et, 'session_key' : session_key_race})
    telemetry_data = pd.DataFrame()
    
    for driver_code, driver_number in driver_config['driver_number'].items():
        try:
            merged_data = utils.merge_data_channels(car_data[car_data["driver_number"]==driver_number].sort_values(by="date"), location_data[location_data["driver_number"]==driver_number].sort_values(by="date"))
            merged_data['distance_l2'] = merged_data.apply(lambda row: utils.compute_l2((row.x, row.y), start_line, before_start_line, after_start_line), axis = 1)/10
            merged_data['distance_regr'] = knn.predict(np.asarray(merged_data[['x', 'y']]))
            merged_data['actual_distance'] = merged_data.apply(lambda row: utils.get_best_distance(row.distance_l2, row.distance_regr, thresh, circuit_length), axis = 1)
            merged_data.reset_index(inplace=True, drop=True)
            
            continuity_counter = 0
            for ind in merged_data.index[1:]:
                if merged_data.loc[ind, 'actual_distance'] - merged_data.loc[ind - (continuity_counter+1), 'actual_distance'] > 2000:
                  merged_data.drop([ind], inplace=True)
                  print(f'Deleted datapoints in {driver_code}s Lap{data["lap_number"][driver_code]}')
                  continuity_counter += 1
                else:
                  continuity_counter = 0
            
            merged_data.reset_index(inplace=True,drop=True)
            merged_data['lap_number'] = utils.assign_lap_number(merged_data, data['lap_number'][driver_code][0], circuit_length, data['lap_number'][driver_code][1])
            telemetry_data = pd.concat([telemetry_data, merged_data])
            data['lap_number'][driver_code][0] = merged_data.iloc[-1].lap_number
            data['lap_number'][driver_code][1] = merged_data.iloc[-1].actual_distance
        except Exception as e:
            print(f'{driver_number} failed')
            print(f'{e} exception')

    telemetry_data.to_sql('telemetry', engine, if_exists = 'append', index = False)
    # print(et, time.time() - t1, sorted(data['lap_number'].items(), key = lambda kv: starting_grid[driver_config['driver_code'][kv[0]]])) # i somehow broke this while making driver_config, fix please
    print(et, time.time() - t1, sorted(data['lap_number'].items()))
    # time.sleep(10)
    # st = max(pd.to_datetime(car_data["date"].iloc[-1]), pd.to_datetime(location_data["date"].iloc[-1]))