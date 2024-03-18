# Import necessary libraries
import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import dash_table
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from urllib.request import urlopen
import json
from scipy.ndimage.filters import gaussian_filter1d

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

def get_lap_dur(driv,lap):
  response = urlopen(f'https://api.openf1.org/v1/laps?session_key=9472&driver_number={driv}&lap_number={lap}')
  return json.loads(response.read().decode('utf-8'))[0]["lap_duration"]

columns = ['driver_code'] 
# columns += [str(lap) for lap in range(1, 11)] 

# Define your Dash app
app = dash.Dash(__name__)


driver1_buttons = html.Div([dbc.Button(f'{driver_config_reverse[driver_number]}', id=f"d-btn1-{driver_number}", color="secondary") for driver_number in driver_config_reverse.keys()],
                    className="driver1-grid gap-2",)
driver2_buttons = html.Div([dbc.Button(f'{driver_config_reverse[driver_number]}', id=f"d-btn2-{driver_number}", color="secondary") for driver_number in driver_config_reverse.keys()],
                    className="driver2-grid gap-2",)
lap1_buttons = html.Div([dbc.Button(f'{lap_number}', id=f"l-btn1-{lap_number}", color="secondary") for lap_number in range(0, 59)],
                    className="lap1-grid gap-2",)
lap2_buttons = html.Div([dbc.Button(f'{lap_number}', id=f"l-btn2-{lap_number}", color="secondary") for lap_number in range(0, 59)],
                    className="lap2-grid gap-2",)



# Define the layout of your app
app.layout = html.Div([
    html.H1("Laptime Comparison for session 9472"),

    dbc.Row(
            [
              dbc.Col([driver1_buttons], md=4),
              dbc.Col([lap1_buttons], md=4),
              dbc.Col([driver2_buttons], md=4),
              dbc.Col([lap2_buttons], md=4),
            ],
            align="right",
        ),
    
    # Dropdown to select table
   
    html.Label("Driver 1"),
    dcc.Dropdown(
        id='driver1-input',
        options=[
            {'label': col, 'value': col} for col in driver_config.keys()
        ],
        style = {'width':'100px'},
        value='VER'  # Default selected column
    ),

    html.Label("Lap 1"),
    dcc.Input(id='lap1-input', type='number', value=1),

    html.Label("Driver 2"),
        dcc.Dropdown(
        id='driver2-input',
        options=[
            {'label': col, 'value': col} for col in driver_config.keys()
        ],
        style = {'width':'100px'},
        value='HAM'  # Default selected column
    ),

    html.Label("Lap 2"),
    dcc.Input(id='lap2-input', type='number', value=1),
    dcc.Interval(
        id='telemetry-updater-component',
        interval=5000,  # in milliseconds
        n_intervals=0
    ),
    dcc.Interval(
        id='weather-updater-component',
        interval=30000,  # in milliseconds
        n_intervals=0
    ),
    dcc.Interval(
        id='laptime-updater-component',
        interval=2000,  # in milliseconds
        n_intervals=0
    ),
    dcc.Interval(
        id='maxspeed-updater-component',
        interval=2000,  # in milliseconds
        n_intervals=0
    ),
    
    
    # Display scatter plot based on the selected table and columns
    dcc.Graph(id='scatter-plot'),
    dcc.Graph(id='laptime-plot'),
    dash_table.DataTable(
        id='maxspeed-table',
        columns=[
            {'name': col, 'id': col} for col in columns
        ],
        fill_width=False,
      # data = pd.DataFrame(columns = columns).to_dict('records')
    ),
    dcc.Graph(id='weather-plot'),
])

# Define callback to update the displayed scatter plot based on the selected table and columns
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('driver1-input', 'value'),
     Input('lap1-input', 'value'),
     Input('driver2-input', 'value'),
     Input('lap2-input', 'value'),
     Input('telemetry-updater-component', 'n_intervals')]
)
def update_scatter_plot(driver1, lap1_number, driver2, lap2_number, n_intervals):
    driver1_number = driver_config[driver1.upper()]
    driver2_number = driver_config[driver2.upper()]
    query = f"SELECT * FROM telemetry WHERE driver_number = '{driver1_number}' and lap_number = '{lap1_number}';"
    df1 = pd.read_sql_query(query, engine)
    df1['date'] = pd.to_datetime(df1.date, format='ISO8601')
    df1[['rpm', 'speed','n_gear', 'throttle', 'drs', 'brake']] = df1[['rpm', 'speed', 'n_gear', 'throttle', 'drs', 'brake']].astype(int)

    query = f"SELECT * FROM telemetry WHERE driver_number = '{driver2_number}' and lap_number = '{lap2_number}';"
    df2 = pd.read_sql_query(query, engine)
    df2['date'] = pd.to_datetime(df2.date, format='ISO8601')
    df2[['rpm', 'speed','n_gear', 'throttle', 'drs', 'brake']] = df2[['rpm', 'speed', 'n_gear', 'throttle', 'drs', 'brake']].astype(int)

    dist1 = gaussian_filter1d(df1.actual_distance, sigma = 10)
    dist2 = gaussian_filter1d(df2.actual_distance, sigma = 10)

    # time1 = df1['date'] - df1['date'].iloc[0]
    # time2 = df2['date'] - df2['date'].iloc[0]
    
    speeds = [go.Scatter(x=dist1, y=df1['speed'], mode='lines', name=f'{driver1.upper()}'),
              go.Scatter(x=dist2, y=df2['speed'], mode='lines', name=f'{driver2.upper()}')]

    throttles = [go.Scatter(x=dist1, y=df1['throttle'], mode='lines', name=f'{driver1.upper()} Thr'),
              go.Scatter(x=dist2, y=df2['throttle'], mode='lines', name=f'{driver2.upper()} Thr'),]
    brakes = [go.Scatter(x=dist1, y=df1['brake'], mode='lines', name=f'{driver1.upper()} Br'),
              go.Scatter(x=dist2, y=df2['brake'], mode='lines', name=f'{driver2.upper()} Br')]

    rpms = [go.Scatter(x=dist1, y=df1['rpm'], mode='lines', name=f'{driver1.upper()}'),
              go.Scatter(x=dist2, y=df2['rpm'], mode='lines', name=f'{driver2.upper()}')]

    gears = [go.Scatter(x=dist1, y=df1['n_gear'], mode='lines', name=f'{driver1.upper()}'),
              go.Scatter(x=dist2, y=df2['n_gear'], mode='lines', name=f'{driver2.upper()}')]

    drss = [go.Scatter(x=dist1, y=df1['drs'], mode='lines', name=f'{driver1.upper()}'),
              go.Scatter(x=dist2, y=df2['drs'], mode='lines', name=f'{driver2.upper()}')]

    fig = make_subplots(rows=6, cols=1, vertical_spacing = 0.01)
    
    for trace in speeds:
        fig.add_trace(trace, row=1, col=1)
    for trace in throttles:
        fig.add_trace(trace, row=2, col=1)
    for trace in brakes:
        fig.add_trace(trace, row=3, col=1)
    for trace in rpms:
        fig.add_trace(trace, row=4, col=1)
    for trace in gears:
        fig.add_trace(trace, row=5, col=1)
    for trace in drss:
        fig.add_trace(trace, row=6, col=1)
        
    fig.update_layout(height=1000, width=1175, title_text=f'''{driver1.upper()} : {get_lap_dur(driver1_number, lap1_number)}s, {driver2.upper()}: {get_lap_dur(driver2_number,lap2_number)}s''')
    
    return fig

@app.callback(
    Output('laptime-plot', 'figure'),
    [Input('laptime-updater-component', 'n_intervals')]
)
def update_laptime_plot(n_intervals):
    # Replace this with your data update logic

    query = f"SELECT driver_number, lap_number, lap_duration FROM laptimes"
    df = pd.read_sql_query(query, engine).dropna().astype(float)

    # df_ = df.pivot(columns = 'lap_number', index = 'driver_number', values = 'lap_duration').round(3)
    # df_.columns = [int(x) for x in df_.columns]
    # df_ = df_[sorted(df_.columns.tolist())].reset_index()

    traces = []
    for k, v in df.groupby('driver_number'):
      traces.append(go.Scatter(x=v['lap_number'], y=v['lap_duration'], mode='markers+lines', name=f'{driver_config_reverse[int(k)]}'))
    layout = go.Layout(title = f'''Laptime Data''', xaxis=dict(title='Lap Number'), yaxis=dict(title='Time'), uirevision = 8)
    figure = go.Figure(data=traces, layout=layout)
    return figure

@app.callback(
    Output('maxspeed-table', 'columns'),
    [Input('maxspeed-updater-component', 'n_intervals')]
)
def update_max_speed_columns(n_intervals):
    query = f"select max(lap_number) as max_lap_number from telemetry"
    df = pd.read_sql_query(query, engine)
    columns = ['driver_code'] + [str(lap) for lap in range(0, 1 + df.iloc[0].max_lap_number)] 
    columns = [{'name': col, 'id': col} for col in columns]
    # print(columns)
    return columns

@app.callback(
    Output('maxspeed-table', 'data'),
    [Input('maxspeed-updater-component', 'n_intervals')]
)
def update_maxspeed_table(n_intervals):
    # Replace this with your data update logic

    query = f"select driver_number, lap_number, max(speed) as max_speed from telemetry group by driver_number, lap_number"
    df = pd.read_sql_query(query, engine)
    df_ = df.pivot(columns = 'lap_number', index = 'driver_number', values = 'max_speed').round(0)
    df_.columns = [int(x) for x in df_.columns]
    df_ = df_[sorted(df_.columns.tolist())].reset_index()
    df_['driver_code'] = df_.driver_number.map(driver_config_reverse)
    df_.columns = [str(x) for x in df_.columns]
    # print(df_.head(2))

    # traces = []
    # for k, v in df.groupby('driver_number'):
    #   traces.append(go.Scatter(x=v['lap_number'], y=v['lap_duration'], mode='markers+lines', name=f'{driver_config_reverse[int(k)]}'))
    # layout = go.Layout(title = f'''Laptime Data''', xaxis=dict(title='Lap Number'), yaxis=dict(title='Time'), uirevision = 8)
    # figure = go.Figure(data=traces, layout=layout)
    return df_.to_dict('records')
    
@app.callback(
    Output('weather-plot', 'figure'),
    [Input('weather-updater-component', 'n_intervals')]
)
def update_weather_plot(n_intervals):

    query = f"SELECT * FROM weather"
    df = pd.read_sql_query(query, engine)

    cols = ['air_temperature', 'humidity',
       'pressure', 'rainfall', 'track_temperature', 'wind_direction',
       'wind_speed']
    df['date'] = pd.to_datetime(df.date, format='ISO8601')
    df[cols] = df[cols].astype(float)
    
    traces = []
    for col in cols:
        # df[col] = df[col].astype(float)
        traces.append(go.Scatter(x=df['date'], y=df[col], mode='lines', name=f'{col}'))
    layout = go.Layout(title = f'''Weather Data''', xaxis=dict(title='Time'), yaxis=dict(title='Value'))
    figure = go.Figure(data=traces, layout=layout)

    return figure

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port = 8050, host = '0.0.0.0')
