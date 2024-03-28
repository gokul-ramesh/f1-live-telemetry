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

def get_starting_grid(session_key, race_start_time):
    grid = get_data(f'https://api.openf1.org/v1/position?session_key={session_key}&date<={race_start_time}')
    return pd.Series(grid["position"].values,index=grid.driver_number).to_dict()

def get_driver_config(session_key):
    driver_data = get_data(f'https://api.openf1.org/v1/drivers?session_key={session_key}')
    driver_config = {row['name_acronym'] : { 'driver_number' : row['driver_number'], 'driver_colour' : row['team_colour']} for ind, row in driver_data.iterrows()}
    driver_reverse_config = {row['driver_number'] : { 'driver_code' : row['name_acronym'], 'driver_colour' : row['team_colour']} for ind, row in driver_data.iterrows()}
    return driver_config, driver_reverse_config

def delta_time(ref, comp):
    ltime = comp['time'].dt.total_seconds().to_numpy()
    ldistance = comp['actual_distance_smoothed'].to_numpy()
    lap_time = np.interp(ref['actual_distance_smoothed'], ldistance, ltime)
    delta = lap_time - ref['time'].dt.total_seconds()
    return delta

def get_data(url):
  return pd.DataFrame(requests.get(url).json())

def save_knn_pickle(pkl_filename, track_layout_filename, session_key_fp1, session_key_fp2, start_line, before_start_line, after_start_line, LAP_THRESHOLDS = 25):

    if os.path.exists(pkl_filename) == False:
    
        lap_data = pd.concat([get_data(f'''https://api.openf1.org/v1/laps?session_key={session_key_fp1}'''), get_data(f'''https://api.openf1.org/v1/laps?session_key={session_key_fp2}''')]).dropna(subset =['lap_duration'])
        lap_data['date_start'] = pd.to_datetime(lap_data['date_start'],format="ISO8601")
        lap_data['date_end'] = lap_data.apply(lambda x: x.date_start + timedelta(seconds = x.lap_duration), axis = 1)
    
        ref_lap_distances = pd.DataFrame()
      
        print("Building model...")
        for _, lap in tqdm(lap_data.sort_values(by = 'lap_duration').iloc[:LAP_THRESHOLDS].iterrows()):
        
            driver_number = lap.driver_number
            start_time = lap.date_start
            end_time = lap.date_end
            session_key = lap.session_key
            lap_duration = (end_time - start_time).total_seconds()
    
            config = {'start_time' : start_time, 'end_time' : end_time, 'session_key' : session_key, 'driver_number' : driver_number}
            car_data, location_data = get_data_channels(config)
            
            merged = merge_data_channels(car_data, location_data)
            merged = compute_distance(merged, start_time)
            ref_lap_distances = pd.concat([ref_lap_distances, merged[['x','y','distance', 'driver_number']]])
        
        ref_lap_distances.dropna(inplace = True)
        
        if not os.path.exists(track_layout_filename):
            ref_lap_distances[['x','y']].to_csv(track_layout_filename, index=True)
            print(f"Saving track layout...")
        
        knn =  KNeighborsRegressor(n_neighbors = 15, weights = 'distance')
        knn.fit(np.asarray(ref_lap_distances[['x', 'y']]), np.asarray(ref_lap_distances[['distance']]))    
        print("Saving model...")
        
        with open(pkl_filename, 'wb') as file:
            pickle.dump([knn, start_line, before_start_line, after_start_line], file)
        
        print("Done")
    else:
        print(f'{pkl_filename} already exists')
        print('Done')
        

def get_session(location, year):
  return get_data(f"https://api.openf1.org/v1/sessions?location={location}&year={year}")

def compute_distance(df, lap_start_time):
    dt = (df['date'] - datetime.now()).dt.total_seconds().diff()
    dt.iloc[0] = (df['date'].iloc[0] - pd.to_datetime(lap_start_time, format='ISO8601')).total_seconds()
    ds = df['speed'] / 3.6 * dt
    df['distance'] = ds.cumsum()
    return df
    
def get_data_channels(config):

    attr_config = {
    'start_time' : 'date>=',
    'end_time' : 'date<=',
    'session_key' : 'session_key=',
    'driver_number' : 'driver_number='
    }
    
    # example config
    # config = {
    #   'start_time' : '2023-04-02T05:00:00',
    #   'end_time' : '2023-04-02T07:00:00',
    #   'session_key' : 7787,
    #   'driver_number' : 1
    # }
    
    car_url = f'''https://api.openf1.org/v1/car_data?'''
    location_url = f'''https://api.openf1.org/v1/location?'''
    query = ''.join([f'''&{attr_config[k]}{v}'''  for k, v in config.items()])
    car_data = get_data(car_url + query)
    location_data = get_data(location_url + query)
    car_data['date'] = pd.to_datetime(car_data['date'], format='ISO8601')
    location_data['date'] = pd.to_datetime(location_data['date'], format='ISO8601')
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
