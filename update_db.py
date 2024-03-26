import requests
import pandas as pd
from sqlalchemy import create_engine, MetaData, delete
import numpy as np
from scipy.stats import linregress as fit

import time
import pickle
import os
from tqdm import tqdm

from datetime import timedelta, datetime
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

def get_data(url):
  return pd.DataFrame(requests.get(url).json())

def get_session(country, year):
  return get_data(f"https://api.openf1.org/v1/sessions?country_name={country}&year={year}")

#Session and circuit information
country = "Australia"
year = 2023
print(f"Getting {country} GP {year}")
session = get_session(country, year)
session_key = session.session_key.iloc[-1]
session_key_FP1 = session.session_key.iloc[0]
session_key_FP2 = session.session_key.iloc[1]
ses_start_time = pd.to_datetime(session.date_start.iloc[-1])
print(f"Event starts at {ses_start_time}")
ses_end_time = pd.to_datetime(session.date_end.iloc[-1])+timedelta(minutes=40)
circuit_length = 5200
before_start_line = (-800, -1700) #Australia
after_start_line = (-1200, -1300)#Australia

driver_config = {row['name_acronym']:row['driver_number'] for ind, row in get_data(f'https://api.openf1.org/v1/drivers?session_key={session_key_FP1+3}').iterrows()}
'''
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
'''
driver_config_reverse = {v: k for k, v in driver_config.items()}


def compute_distance(df, lap_start_time):
    dt = (df['date'] - datetime.now()).dt.total_seconds().diff()
    dt.iloc[0] = (df['date'].iloc[0] - pd.to_datetime(lap_start_time, format='ISO8601')).total_seconds()
    ds = df['speed'] / 3.6 * dt
    df['distance'] = ds.cumsum()
    
def get_data_channels(session_key, start_time, end_time):
  url = f'''https://api.openf1.org/v1/car_data?&session_key={session_key}&date<{end_time}&date>={start_time}'''
  car_data = get_data(url)
  url = f'''https://api.openf1.org/v1/location?&session_key={session_key}&date<{end_time}&date>={start_time}'''
  location_data = get_data(url)

  return car_data, location_data

def get_weather_data(session_key, start_time, end_time):
  url = f'''https://api.openf1.org/v1/weather?&session_key={session_key}&date<{end_time}&date>={start_time}'''
  weather = get_data(url)
  if len(weather):
    # weather['date'] = pd.to_datetime(weather['date'], format='ISO8601')
    return weather
  else:
    return pd.DataFrame()

def get_position_data(session_key, end_time):
  url = f'''https://api.openf1.org/v1/position?session_key={session_key}&date<={end_time}'''
  position = get_data(url)
  position = position[['driver_number', 'date', 'position']].groupby('driver_number').agg('last').reset_index()
  if len(position):
    # weather['date'] = pd.to_datetime(weather['date'], format='ISO8601')
    return position
  else:
    return pd.DataFrame()


def get_laptimes_data(session_key, start_time, end_time):
  url = f'''https://api.openf1.org/v1/laps?&session_key={session_key}&date_start<{end_time}&date_start>={start_time}'''
  laptimes = get_data(url)
  if len(laptimes):
    # laptimes['date_start'] = pd.to_datetime(laptimes['date_start'], format='ISO8601')
    return laptimes
  else:
    return pd.DataFrame()


def merge_data_channels(car_data, location_data):

  merged = pd.merge(car_data, location_data, how = 'outer', on = ['date', 'meeting_key', 'session_key', 'driver_number']).sort_values(by = 'date')
  merged['date'] = pd.to_datetime(merged['date'], format='ISO8601')
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

def assign_lap_number(data, current_lap, circuit_length, latest_dist):
  trans_dist = data.iloc[0].actual_distance - latest_dist
  return np.where(-data['actual_distance'].diff().fillna(trans_dist) > 0.90 * circuit_length, 1, 0).cumsum() + current_lap

def compute_l2(car_location, start_line, before_start_line, after_start_line):

  # can do the same with just after_start_line & start_line, before_start_line unnecessary

  track_dir_vector = np.array([after_start_line[0] - before_start_line[0], after_start_line[1] - before_start_line[1]])
  track_car_vector = np.array([car_location[0] - start_line[0], car_location[1] - start_line[1]])
  s = (track_car_vector[0]**2 + track_car_vector[1]**2)**0.5
  return np.sign(np.dot(track_dir_vector, track_car_vector)) * s

pkl_filename = f"knn_{country}-{year}_FP1_FP2_top25.pkl"
print(pkl_filename)


# make knn


if os.path.exists(pkl_filename) == False:

  url = f'''https://api.openf1.org/v1/laps?session_key={session_key_FP1}'''
  lap_data = get_data(url)
  lap_data = lap_data[lap_data.lap_duration.notna()]
  # display(lap_data)
  lap_data['date_start'] = pd.to_datetime(lap_data['date_start'],format="mixed")
  lap_data['date_end'] = lap_data.apply(lambda x: x.date_start + timedelta(seconds = x.lap_duration), axis = 1)
  
  url = f'''https://api.openf1.org/v1/laps?session_key={session_key_FP2}'''
  lap_data2 = get_data(url)
  lap_data2 = lap_data2[lap_data2.lap_duration.notna()]
  # display(lap_data)
  lap_data2['date_start'] = pd.to_datetime(lap_data2['date_start'], format='mixed')
  lap_data2['date_end'] = lap_data2.apply(lambda x: x.date_start + timedelta(seconds = x.lap_duration), axis = 1)
  lap_data = pd.concat([lap_data, lap_data2])
  top_laps = lap_data.sort_values(by='lap_duration').iloc[:50, :]
  
  
  ref_lap_distances = pd.DataFrame()
  num_laps = 0
  start_line_dp = pd.DataFrame()
  
  LAP_THRESHOLDS = 25
  
  print("Building model...")
  # for _, lap in lap_data.sort_values(by = 'lap_duration').iterrows():
  for _, lap in tqdm(lap_data.sort_values(by = 'lap_duration').iterrows(), total = LAP_THRESHOLDS):

      if num_laps > LAP_THRESHOLDS:
          break
      # print(f"{num_laps * 4}% complete ...")
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
      start_line_dp=pd.concat([start_line_dp,merged.iloc[[1,2,3,4,5,6,7,8,9,10,0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10],:]])
  
  ref_lap_distances.dropna(inplace = True)
  
  if not os.path.exists(f"track_layout/{country}-{year}.csv"):
    ref_lap_distances[['x','y']].to_csv(f'track_layout/{country}-{year}.csv', index=True)
    print(f"Saving track layout...")

  knn =  KNeighborsRegressor(n_neighbors = 15, weights = 'distance')
  knn.fit(np.asarray(ref_lap_distances[['x', 'y']]), np.asarray(ref_lap_distances[['distance']]))
  
  #Find starting line
  start_line_dp.dropna(inplace=True)
  start_line = fit(start_line_dp.x, start_line_dp.y)
  start_line_dp['distance'] = np.where(start_line_dp['distance']>5000, start_line_dp['distance']-circuit_length, start_line_dp['distance'])
  start_lines = pd.DataFrame(columns=['x','y'])
  start_line_dp.drop(start_line_dp[(start_line_dp['x']==0) & (start_line_dp['y']==0)].index, inplace=True)
  start_line_dp["start_coords_x"] = start_line_dp.x + start_line_dp['distance']/(1+start_line.slope)**0.5
  start_line_dp["start_coords_y"] = start_line_dp.y + start_line.slope * start_line_dp['distance']/(1+start_line.slope)**0.5
  
  start_line = (start_line_dp['start_coords_x'].mean(), start_line_dp['start_coords_y'].mean())
  print("Saving model...")

  with open(pkl_filename, 'wb') as file:
      pickle.dump([knn, start_line, before_start_line, after_start_line], file)

with open(pkl_filename, 'rb') as file:
    knn, start_line, before_start_line, after_start_line = pickle.load(file)

grid = get_data(f'https://api.openf1.org/v1/position?session_key={session_key}&date<={ses_start_time}')
# grid[['driver_number', 'position']].to_dict('index')
starting_grid = pd.Series(grid["position"].values,index=grid.driver_number).to_dict()
#starting_grid

# Connect to your SQL database
db_file = f"{session_key}.db"
if os.path.isfile(db_file):
    os.remove(db_file)
    print(f"Removing older DB file {db_file}")
print(f"Connecting to db using {db_file}")
engine = create_engine(f"sqlite:///{db_file}")


thresh = 100

interval = 300

lap_number,latest_distance = {}, {}
for driver_code, driver_number in driver_config.items():
  lap_number[driver_code] = 0
  latest_distance[driver_code] = np.nan

#for timestamp in pd.date_range(start_time, end_time, freq = f'{interval}s'):
st = ses_start_time + timedelta(seconds=30)

while True:
  et = st + timedelta(seconds=interval)
  print(st,et)
  if et>ses_end_time:
    break

  t1 = time.time()
  

  # try:
  weather_data = get_weather_data(session_key, st, et)
  laptimes_data = get_laptimes_data(session_key, st, et)
  position_data = get_position_data(session_key, et)
  if len(weather_data):
    weather_data.map(str).to_sql('weather', engine, if_exists = 'append', index = False)
  if len(laptimes_data):
    laptimes_data.map(str).to_sql('laptimes', engine, if_exists = 'append', index = False)
  if len(position_data):
    position_data.map(str).to_sql('position', engine, if_exists = 'replace', index = False)
  # except Exception as e:
  #   print(f'{driver_number} failed')
  #   print(f'{e} exception')

  car_data, location_data = get_data_channels(session_key, st, et)
  telemetry_data = pd.DataFrame()
    
  for driver_code, driver_number in driver_config.items():
    try:
      merged_data = merge_data_channels(car_data[car_data["driver_number"]==driver_number].sort_values(by="date"), location_data[location_data["driver_number"]==driver_number].sort_values(by="date"))
      merged_data['distance_l2'] = merged_data.apply(lambda row: compute_l2((row.x, row.y), start_line, before_start_line, after_start_line), axis = 1)/10
      merged_data['distance_regr'] = knn.predict(np.asarray(merged_data[['x', 'y']]))
      merged_data['actual_distance'] = merged_data.apply(lambda row: get_best_distance(row.distance_l2, row.distance_regr, thresh, circuit_length), axis = 1)
      merged_data.reset_index(inplace=True, drop=True)
      continuity_counter = 0
      merged_data = pd.concat([pd.DataFrame({"actual_distance":[latest_distance[driver_code]]}), merged_data], ignore_index=True)
      for ind in merged_data.index[1:]:
        if merged_data.loc[ind, 'actual_distance'] - merged_data.loc[ind - (continuity_counter+1), 'actual_distance'] > 2000:
          merged_data.drop([ind], inplace=True)
          continuity_counter += 1
          print(f'Deleted datapoints in {driver_code}s Lap{data["lap_number"][driver_code]}')
        else:
          continuity_counter = 0
      merged_data.reset_index(inplace=True,drop=True)
      merged_data['lap_number'] = assign_lap_number(merged_data, lap_number[driver_code], circuit_length, latest_distance[driver_code])
      telemetry_data = pd.concat([telemetry_data, merged_data])
      lap_number[driver_code] = merged_data.iloc[-1].lap_number
      latest_distance[driver_code] = merged_data.iloc[-1].actual_distance
    except Exception as e:
      print(f'{driver_number} failed')
      print(f'{e} exception')

  telemetry_data.to_sql('telemetry', engine, if_exists = 'append', index = False)
  print(et, time.time() - t1, sorted(lap_number.items(), key = lambda kv: starting_grid[driver_config[kv[0]]]))
  st = max(pd.to_datetime(car_data["date"].iloc[-1]), pd.to_datetime(location_data["date"].iloc[-1]))
  # break



