import requests
import pandas as pd
from sqlalchemy import create_engine, MetaData, delete
import numpy as np

import time
import pickle

from datetime import timedelta, datetime
from sklearn.neighbors import KNeighborsRegressor

# Connect to your SQL database
engine = create_engine("sqlite:///9472.db")

driver_config = {'VER': 1,
  'SAR': 2,
  'RIC': 3,
  'NOR': 4,
  'GAS': 10,
  'PER': 11,
  'ALO': 14,
  'LEC': 16,
  'STR': 18,
  'MAG': 20,
  'TSU': 22,
  'ALB': 23,
  'ZHO': 24,
  'HUL': 27,
  'OCO': 31,
  'HAM': 44,
  'SAI': 55,
  'RUS': 63,
  'BOT': 77,
  'PIA': 81}

driver_config_reverse = {v: k for k, v in driver_config.items()}

def get_data(url):
  return pd.DataFrame(requests.get(url).json())

def compute_distance(df, lap_start_time):
    dt = (df['date'] - datetime.now()).dt.total_seconds().diff()
    dt.iloc[0] = (df['date'].iloc[0] - pd.to_datetime(lap_start_time)).total_seconds()
    ds = df['speed'] / 3.6 * dt
    df['distance'] = ds.cumsum()
    
def get_data_channels(session_key, start_time, end_time):
  url = f'''https://api.openf1.org/v1/car_data?&session_key={session_key}&date<{end_time}&date>={start_time}'''
  car_data = get_data(url)
  url = f'''https://api.openf1.org/v1/location?&session_key={session_key}&date<{end_time}&date>={start_time}'''
  location_data = get_data(url)

  return car_data, location_data


def merge_data_channels(car_data, location_data):

  merged = pd.merge(car_data, location_data, how = 'outer', on = ['date', 'meeting_key', 'session_key', 'driver_number']).sort_values(by = 'date')
  merged['date'] = pd.to_datetime(merged['date'])
  merged.set_index('date', inplace = True)
  merged[['n_gear', 'drs']] = merged[['n_gear', 'drs']].ffill().ffill().bfill()
  merged[['rpm', 'speed', 'throttle', 'brake']] = merged[['rpm', 'speed', 'throttle', 'brake']].interpolate(method = 'polynomial', order = 1, limit_direction = 'both')
  merged[['x', 'y', 'z']] = merged[['x', 'y', 'z']].interpolate(method = 'polynomial', order = 2, limit_direction = 'both')
  merged.reset_index(inplace = True)
  merged.dropna(inplace = True)

  return merged
  
def get_best_distance(l2, regr, thresh, circuit_length):
  if abs(l2) < thresh:
     return l2 if l2 > 0 else circuit_length + l2
  else:
    return regr

def assign_lap_number(data, current_lap, circuit_length):
  return np.where(-data['actual_distance'].diff() > 0.90 * circuit_length, 1, 0).cumsum() + current_lap

def compute_l2(car_location, start_line, before_start_line, after_start_line):

  # can do the same with just after_start_line & start_line, before_start_line unnecessary

  track_dir_vector = np.array([after_start_line[0] - before_start_line[0], after_start_line[1] - before_start_line[1]])
  track_car_vector = np.array([car_location[0] - start_line[0], car_location[1] - start_line[1]])
  s = (track_car_vector[0]**2 + track_car_vector[1]**2)**0.5
  return np.sign(np.dot(track_dir_vector, track_car_vector)) * s
"""  
# make knn

url = f'''https://api.openf1.org/v1/laps?session_key=9465'''
lap_data = get_data(url)
lap_data = lap_data[lap_data.lap_duration.notna()]
# display(lap_data)
lap_data['date_start'] = pd.to_datetime(lap_data['date_start'])
lap_data['date_end'] = lap_data.apply(lambda x: x.date_start + timedelta(seconds = x.lap_duration), axis = 1)

url = f'''https://api.openf1.org/v1/laps?session_key=9466'''
lap_data2 = get_data(url)
lap_data2 = lap_data2[lap_data2.lap_duration.notna()]
# display(lap_data)
lap_data2['date_start'] = pd.to_datetime(lap_data2['date_start'], format='mixed')
lap_data2['date_end'] = lap_data2.apply(lambda x: x.date_start + timedelta(seconds = x.lap_duration), axis = 1)
lap_data = pd.concat([lap_data, lap_data2])

ref_lap_distances = pd.DataFrame()
num_laps = 0

for _, lap in lap_data.sort_values(by = 'lap_duration').iterrows():

    if num_laps > 25:
        break

    driver_number = lap.driver_number
    start_time = lap.date_start
    end_time = lap.date_end
    session_key = lap.session_key
    lap_duration = (end_time - start_time).total_seconds()

    if lap_duration > 98:
        continue
    num_laps +=1
    # print(num_laps, lap_duration, driver_number, start_time)

    url = f'''https://api.openf1.org/v1/car_data?driver_number={driver_number}&session_key={session_key}&date<{end_time}&date>={start_time}'''
    car_data = get_data(url)
    url = f'''https://api.openf1.org/v1/location?driver_number={driver_number}&session_key={session_key}&date<{end_time}&date>={start_time}'''
    location_data = get_data(url)

    merged = pd.merge(car_data, location_data, how = 'outer', on = ['date', 'meeting_key', 'session_key', 'driver_number']).sort_values(by = 'date')
    merged['date'] = pd.to_datetime(merged['date'])
    merged.set_index('date', inplace = True)
    merged[['n_gear', 'drs']] = merged[['n_gear', 'drs']].ffill().ffill().bfill()
    merged[['rpm', 'speed', 'throttle', 'brake']] = merged[['rpm', 'speed', 'throttle', 'brake']].interpolate(method = 'polynomial', order = 1, limit_direction = 'both')
    merged[['x', 'y', 'z']] = merged[['x', 'y', 'z']].interpolate(method = 'polynomial', order = 2, limit_direction = 'both')
    merged.reset_index(inplace = True)

    compute_distance(merged, start_time)
    ref_lap_distances = pd.concat([ref_lap_distances, merged[['x','y','distance', 'driver_number']]])

ref_lap_distances.dropna(inplace = True)

knn =  KNeighborsRegressor(n_neighbors = 15, weights = 'distance')
knn.fit(np.asarray(ref_lap_distances[['x', 'y']]), np.asarray(ref_lap_distances[['distance']]))


pkl_filename = "knn_Bahrain_FP1_FP2_top25.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(knn, file)

"""
pkl_filename = "knn_Bahrain_FP1_FP2_top25.pkl"
with open(pkl_filename, 'rb') as file:
    knn = pickle.load(file)
    
grid = get_data('https://api.openf1.org/v1/position?session_key=9472&date<=2024-03-02T15:00:00')
# grid[['driver_number', 'position']].to_dict('index')
starting_grid = pd.Series(grid.position.values,index=grid.driver_number).to_dict()
#starting_grid

start_line = (-371.4, 1462.6)
before_start_line = (-400, 873)
after_start_line = (-330, 2433)

start_time = '2024-03-02T15:00:30.000000'
end_time = '2024-03-02T16:10:00.000000'

session_key = 9472

thresh = 100
circuit_length = 5412

interval = 30

data = {}
data['data'] = {}
data['lap_number'] = {}
for driver_code, driver_number in driver_config.items():
  data['data'][driver_number] = pd.DataFrame()
  data['lap_number'][driver_number] = 0

for timestamp in pd.date_range(start_time, end_time, freq = f'{interval}s'):
  st = timestamp
  et = timestamp + timedelta(seconds = interval)
  # print(st)
  #t1 = time.time()
  retries = 0
  while True or retries<10:
    try:
      car_data, location_data = get_data_channels(session_key, st, et)
    except Exception as e:
      retries += 1
      print(e)
      print("Retrying ...")
      continue
    break
   
  if retries>10:
    break
  #print(time.time()-t1)
  for driver_code, driver_number in driver_config.items():
    
    
    try:
      #t2 = time.time()
      merged_data = merge_data_channels(car_data.loc[car_data["driver_number"]==driver_number], location_data.loc[location_data["driver_number"]==driver_number]).sort_values(by="date")
      #t3 = time.time()
      #print(f"Merge channels : {t3-t2}")
      merged_data['distance_l2'] = merged_data.apply(lambda row: compute_l2((row.x, row.y), start_line, before_start_line, after_start_line), axis = 1)/10
      merged_data['distance_regr'] = knn.predict(np.asarray(merged_data[['x', 'y']]))
      merged_data['actual_distance'] = merged_data.apply(lambda row: get_best_distance(row.distance_l2, row.distance_regr, thresh, circuit_length), axis = 1)
      merged_data['lap_number'] = assign_lap_number(merged_data, data['lap_number'][driver_number], circuit_length)
      #t4 = time.time()
      #print(f"Prediction : {t4-t3}")
      # data['data'][driver_number] = pd.concat([data['data'][driver_number], merged_data])
      merged_data.to_sql('telemetry', engine, if_exists = 'append', index = False)
      #t5 = time.time()
      #print(f"Write to SQL : {t5-t4}")
      data['lap_number'][driver_number] = merged_data.iloc[-1].lap_number
    except Exception as e:
      print(f'{driver_number} failed')
      print(f'{e} exception')

    # add distance logic
    # t3 = time.time()
    # print(driver_code, t3-t2)
  # t4 = time.time()
  # print('all drivers', t4-t1)
  # print(et, data['lap_number'])
  print(et, sorted(data['lap_number'].items(), key = lambda kv: starting_grid[kv[0]]))
  # break



