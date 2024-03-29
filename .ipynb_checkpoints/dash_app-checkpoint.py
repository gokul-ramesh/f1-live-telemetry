# Import necessary libraries
import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import dash_table
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from urllib.request import urlopen
import json
from scipy.ndimage import gaussian_filter1d, uniform_filter
import requests
import utils
import sys

# Bahrain 
# corners = [711.9508171931816, 814.1876569292108, 936.7201493049793, 1506.9841172483643, 1787.9086361225473, 1880.658880160338, 1975.1619299965303, 2232.6789270606096, 2598.3923945375827, 2691.728150228979, 3466.886530179646, 3875.788302688632, 4084.7641184324057, 4889.970687598453, 4970.110027289251]
# corners = [ 354.42083043,  439.27690256, 1080.70076863, 1232.24836987,
#        1432.90920875, 1851.34924493, 1950.29260385, 2152.99296282,
#        3258.34533469, 3367.69508844, 4078.44514627, 4341.73730817,
#        4580.99734427, 4742.19833527] #Australia
# # Azerbaijan
# corners = [840.4395031422582, 925.2192705531841, 1168.2113212093552, 1741.685572018549, 2122.736856995752, 2364.579752964712, 2547.5431414033974, 2637.1095897460928, 2925.3814319017774, 3499.034858626688, 3638.214025589213, 3781.746397745988, 4025.8641801250633, 4156.471023673764, 4203.5096385535135, 4371.906121536575]

#Session and circuit information

TOTAL_LAPS = 75
track_config = pd.read_csv('config/track_config.csv')

location = sys.argv[1]
year = int(sys.argv[2])
needed_session = sys.argv[3]

track = track_config.query(f''' circuit_location == '{location}' ''')


circuit_length = int(track.circuit_length.iloc[0])
corners = eval(track.corners.iloc[0])
start_line = eval(track.start_line.iloc[0])
before_start_line = eval(track.before_start_line.iloc[0])
after_start_line = eval(track.after_start_line.iloc[0])

session = utils.get_session(location, year)
session_key = session.query(f" session_name == '{needed_session}'").session_key.iloc[0]

# Connect to your SQL database
engine = create_engine(f"sqlite:///data/{session_key}.db")
print(f"Loading from data/{session_key}.db")

driver_data = pd.read_csv(f'config/driver_config_{year}.csv')
# driver_color = {}
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

lines = {}
for name, color in driver_config['team_colour'].items():
  lines[driver_config['driver_number'][name]] = dict(color=f"{color}")
  if driver_config['driver_number'][name] in [11, 18, 22, 23, 27, 31, 55, 63, 77, 81]:
    lines[driver_config['driver_number'][name]]['dash'] = 'dot'

# driver_config_reverse = {v: k for k, v in driver_config.items()}

def get_lap_dur(driv,lap):
  response = urlopen(f'https://api.openf1.org/v1/laps?session_key={session_key}&driver_number={driv}&lap_number={lap}')
  return json.loads(response.read().decode('utf-8'))[0]["lap_duration"]

columns = ['driver_code'] 
# columns += [str(lap) for lap in range(1, 11)] 

# Define your Dash app
# app = dash.Dash(__name__)
app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])


driver1_button_group = html.Div(
    [
        dbc.RadioItems(
            id="driver1-radiobuttons",
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-secondary",
            labelCheckedClassName="active",
                options=[{'label': html.Div([key], style={'color': 'Black', 'font-size': 20, 'text-align': 'center'}), 'value':key} for key, value in driver_config['driver_number'].items()],
            value='VER',
          # labelStyle= {"margin":"0.001rem"}
        ),
        html.Div(id="output"),
    ],
    className="radio-group",
)

driver2_button_group = html.Div(
    [
        dbc.RadioItems(
            id="driver2-radiobuttons",
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-secondary",
            labelCheckedClassName="active",
                options=[{'label': html.Div([key], style={'color': 'Black', 'font-size': 20, 'text-align': 'center'}), 'value':key} for key, value in driver_config['driver_number'].items()],
            value='LEC',
          # labelStyle= {"margin":"0.001rem"}
        ),
        html.Div(id="output"),
    ],
    className="radio-group",
)

lap1_button_group = html.Div(
    [
        dbc.RadioItems(
            id="lap1-radiobuttons",
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-secondary",
            labelCheckedClassName="active",
                options=[{'label': html.Div([lap_number], style={'color': 'Black', 'font-size': 16, 'text-align': 'center'}), 'value':lap_number} for lap_number in range(0, 1 + TOTAL_LAPS)],
          # size="sm",
		value=1,  # Default value
          # labelStyle= {"margin":"0.001rem"}
        ),
        html.Div(id="output"),
        
    ],
    className="radio-group",
)

lap2_button_group = html.Div(
    [
        dbc.RadioItems(
            id="lap2-radiobuttons",
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-secondary",
            labelCheckedClassName="active",
                options=[{'label': html.Div([lap_number], style={'color': 'Black', 'font-size': 16, 'text-align': 'center'}), 'value':lap_number} for lap_number in range(0, 1 + TOTAL_LAPS)],
		value=1,  # Default value
          # labelStyle= {"margin":"0.001rem"}
        ),
        html.Div(id="output"),
    ],
    className="radio-group",
)




# Define the layout of your app
app.layout = html.Div([
    html.H1(f"Laptime Comparison for Race {location}, {year}"),

  # driver1_button_group,
    dbc.Col(
      [
                dbc.Row([driver1_button_group]),
                dbc.Row([lap1_button_group]),
                dbc.Row([driver2_button_group]),
                dbc.Row([lap2_button_group]),
      ], align = 'left'
    ),
    html.Button('Submit', id='telemetry-submit-value', n_clicks=0),
  
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
        interval=10000,  # in milliseconds
        n_intervals=0
    ),
    dcc.Interval(
        id='maxspeed-updater-component',
        interval=2000,  # in milliseconds
        n_intervals=0
    ),
    dcc.Interval(
        id='position-updater-component',
        interval=2000,  # in milliseconds
        n_intervals=0
    ),
    dcc.Interval(
        id='track-location-updater-component',
        interval=2000,  # in milliseconds
        n_intervals=0
    ),
    
    
    # Display scatter plot based on the selected table and columns
    dcc.Graph(id='scatter-plot'),
    dash_table.DataTable(
        id='corner-minspeed-table',
        columns=[
            {'name': str(col), 'id': str(col)} for col in ['driver_code'] + [*range(1, 1+len(corners))]
        ],
        fill_width=False,
      # data = pd.DataFrame(columns = columns).to_dict('records')
    ),
    dcc.Input(id="laptime-threshold-input", type="number", placeholder="", size = '5px', step=1, value = 200),
    dcc.Graph(id='laptime-plot'),
    dash_table.DataTable(
        id='maxspeed-table',
        columns=[
            {'name': col, 'id': col} for col in columns
        ],
        fill_width=False,
      # data = pd.DataFrame(columns = columns).to_dict('records')
    ),
   dash_table.DataTable(
        id='samples-table',
        columns=[
            {'name': col, 'id': col} for col in columns
        ],
        fill_width=False,
      # data = pd.DataFrame(columns = columns).to_dict('records')
    ),
    dcc.Graph(id='track-location-plot'),
    dash_table.DataTable(
        id='position-table',
        columns=[
            {'name': col, 'id': col} for col in ['driver_code', 'position', 'date', 'lap_number']
        ],
        fill_width=False,
      # data = pd.DataFrame(columns = columns).to_dict('records')
    ),
    dcc.Graph(id='weather-plot'),
])

# Define callback to update the displayed scatter plot based on the selected table and columns
@app.callback(
    Output('scatter-plot', 'figure'),
    [State('driver1-radiobuttons', 'value'),
     State('lap1-radiobuttons', 'value'),
     State('driver2-radiobuttons', 'value'),
     State('lap2-radiobuttons', 'value'),
     Input('telemetry-submit-value', 'n_clicks'),
     Input('telemetry-updater-component', 'n_intervals')]
)
def update_scatter_plot(driver1, lap1_number, driver2, lap2_number, n_clicks, n_intervals):
    driver1_number = driver_config['driver_number'][driver1.upper()]
    driver2_number = driver_config['driver_number'][driver2.upper()]
    query = f"SELECT * FROM telemetry WHERE ((driver_number = '{driver1_number}' and lap_number = '{lap1_number}') or (driver_number = '{driver2_number}' and lap_number = '{lap2_number}'));"
    df = pd.read_sql_query(query, engine)
    df['date'] = pd.to_datetime(df.date, format='ISO8601')
    df[['rpm', 'speed','n_gear', 'throttle', 'drs', 'brake', 'driver_number', 'lap_number']] = df[['rpm', 'speed', 'n_gear', 'throttle', 'drs', 'brake', 'driver_number', 'lap_number']].astype(int)

    df1 = df[(df.lap_number == lap1_number) & (df.driver_number == driver1_number)]
    df2 = df[(df.lap_number == lap2_number) & (df.driver_number == driver2_number)]
  
    # df1 = pd.read_sql_query(query, engine)
    # df1['date'] = pd.to_datetime(df1.date, format='ISO8601')
    # df1[['rpm', 'speed','n_gear', 'throttle', 'drs', 'brake']] = df1[['rpm', 'speed', 'n_gear', 'throttle', 'drs', 'brake']].astype(int)

    # query = f"SELECT * FROM telemetry WHERE driver_number = '{driver2_number}' and lap_number = '{lap2_number}';"
    # df2 = pd.read_sql_query(query, engine)
    # df2['date'] = pd.to_datetime(df2.date, format='ISO8601')
    # df2[['rpm', 'speed','n_gear', 'throttle', 'drs', 'brake']] = df2[['rpm', 'speed', 'n_gear', 'throttle', 'drs', 'brake']].astype(int)

    df1['time'] = df1['date'] - df1['date'].iloc[0]
    df2['time'] = df2['date'] - df2['date'].iloc[0]
    df1['actual_distance_smoothed'] = gaussian_filter1d(df1.actual_distance, sigma = 10)
    df2['actual_distance_smoothed'] = gaussian_filter1d(df2.actual_distance, sigma = 10)
    delta = utils.delta_time(df1, df2)
    smoothed_delta = uniform_filter(delta, size = 30)

    line_driver2 = dict(color=f"{driver_config['team_colour'][driver2.upper()]}")
    if driver_config['team_colour'][driver1.upper()] == driver_config['team_colour'][driver2.upper()]:
        line_driver2['dash'] = "dot"
    # time1 = df1['date'] - df1['date'].iloc[0]
    # time2 = df2['date'] - df2['date'].iloc[0]
    
    speeds = [go.Scatter(x=df1.actual_distance_smoothed, y=df1['speed'], mode='lines', name=f'{driver1.upper()}, Lap {lap1_number}', line=dict(color=f"{driver_config['team_colour'][driver1.upper()]}"), legendgroup='group1'),
              go.Scatter(x=df2.actual_distance_smoothed, y=df2['speed'], mode='lines', name=f'{driver2.upper()}, Lap {lap2_number}', line=line_driver2, legendgroup='group2'),
              go.Scatter(x=df1.actual_distance_smoothed, y=df1['drs']*5, mode='lines', name=f'{driver1.upper()}', line=dict(color=f"{driver_config['team_colour'][driver1.upper()]}"), legendgroup='group1', showlegend=False),
              go.Scatter(x=df2.actual_distance_smoothed, y=df2['drs']*5, mode='lines', name=f'{driver2.upper()}', line=line_driver2, legendgroup='group2', showlegend=False)]

    for corner in corners:
        speeds.append(go.Scatter(x=[corner,corner], y=[0,320], mode='lines', line=dict(color="#404040", dash="dot"), showlegend=False))

    deltas = [go.Scatter(x=df1.actual_distance_smoothed, y=smoothed_delta, mode='lines', showlegend= False, line=dict(color="#404040"))]
    deltas.append(go.Scatter(x=[0, circuit_length], y=[0,0], mode='lines', line=dict(color="#606060", dash = "dash"), showlegend=False))
    for corner in corners:
        deltas.append(go.Scatter(x=[corner,corner], y=[min(smoothed_delta), max(smoothed_delta)], mode='lines', line=dict(color="#404040", dash="dot"), showlegend=False))

    throttles = [go.Scatter(x=df1.actual_distance_smoothed, y=df1['throttle'], mode='lines', name=f'{driver1.upper()}', line=dict(color=f"{driver_config['team_colour'][driver1.upper()]}"), legendgroup='group1',showlegend=False),
                go.Scatter(x=df2.actual_distance_smoothed, y=df2['throttle'], mode='lines', name=f'{driver2.upper()}', line=line_driver2, legendgroup='group2',showlegend=False),]
    for corner in corners:
        throttles.append(go.Scatter(x=[corner,corner], y=[0,100], mode='lines', line=dict(color="#404040", dash="dot"), showlegend=False))

    brakes = [go.Scatter(x=df1.actual_distance_smoothed, y=df1['brake'], mode='lines', name=f'{driver1.upper()}', line=dict(color=f"{driver_config['team_colour'][driver1.upper()]}"), legendgroup='group1',showlegend=False),
              go.Scatter(x=df2.actual_distance_smoothed, y=df2['brake'], mode='lines', name=f'{driver2.upper()}', line=line_driver2, legendgroup='group2',showlegend=False)]
    for corner in corners:
        brakes.append(go.Scatter(x=[corner,corner], y=[0,100], mode='lines', line=dict(color="#404040", dash="dot"), showlegend=False))

    rpms = [go.Scatter(x=df1.actual_distance_smoothed, y=df1['rpm'], mode='lines', name=f'{driver1.upper()}', line=dict(color=f"{driver_config['team_colour'][driver1.upper()]}"), legendgroup='group1',showlegend=False),
              go.Scatter(x=df2.actual_distance_smoothed, y=df2['rpm'], mode='lines', name=f'{driver2.upper()}', line=line_driver2, legendgroup='group2',showlegend=False)]

    for corner in corners:
        rpms.append(go.Scatter(x=[corner,corner], y=[0,12000], mode='lines', line=dict(color="#404040", dash="dot"), showlegend=False))
    
    gears = [go.Scatter(x=df1.actual_distance_smoothed, y=df1['n_gear'], mode='lines', name=f'{driver1.upper()}', line=dict(color=f"{driver_config['team_colour'][driver1.upper()]}"), legendgroup='group1',showlegend=False),
              go.Scatter(x=df2.actual_distance_smoothed, y=df2['n_gear'], mode='lines', name=f'{driver2.upper()}', line=line_driver2, legendgroup='group2',showlegend=False)]

    for corner in corners:
        gears.append(go.Scatter(x=[corner,corner], y=[0,8], mode='lines', line=dict(color="#404040", dash="dot"), showlegend=False))

    # fig = make_subplots(rows=5, cols=1, vertical_spacing = 0.01, row_width = [0.4, 0.15, 0.15, 0.15, 0.15])
    fig = make_subplots(rows=6, cols=1, vertical_spacing = 0.005, row_width = [0.12, 0.12, 0.12, 0.12, 0.22, 0.30]) # don't ask me how this works, the reverse of this row_width list is what I expected to work - it made the last plot the tallest
    
    for trace in speeds:
        fig.add_trace(trace, row=1, col=1)
    for trace in deltas:
        fig.add_trace(trace, row=2, col=1)
    for trace in throttles:
        fig.add_trace(trace, row=3, col=1)
    for trace in brakes:
        fig.add_trace(trace, row=4, col=1)
    for trace in rpms:
        fig.add_trace(trace, row=5, col=1)
    for trace in gears:
        fig.add_trace(trace, row=6, col=1)
    #for trace in drss:
    #    fig.add_trace(trace, row=6, col=1)
       
    fig['layout']['yaxis']['title']="Speed/DRS"
    fig['layout']['yaxis2']['title']="Delta"
    fig['layout']['yaxis3']['title']="Throttle"
    fig['layout']['yaxis4']['title']="Brake"
    fig['layout']['yaxis5']['title']="RPM"
    fig['layout']['yaxis6']['title']="Gear"
    fig.update_layout(uirevision=8, height=1400, width=1800, title_text=f'''{driver1.upper()} : {get_lap_dur(driver1_number, lap1_number)}s, {driver2.upper()}: {get_lap_dur(driver2_number,lap2_number)}s''')
    fig.update_xaxes(showticklabels=False)

    return fig

# Define callback to update the displayed scatter plot based on the selected table and columns
@app.callback(
    Output('corner-minspeed-table', 'data'),
   [State('driver1-radiobuttons', 'value'),
     State('lap1-radiobuttons', 'value'),
     State('driver2-radiobuttons', 'value'),
     State('lap2-radiobuttons', 'value'),
     Input('telemetry-submit-value', 'n_clicks'),
     Input('telemetry-updater-component', 'n_intervals')]
)
def update_corner_minspeed_table(driver1, lap1_number, driver2, lap2_number, n_clicks, n_intervals):

    driver1_number = driver_config['driver_number'][driver1.upper()]
    driver2_number = driver_config['driver_number'][driver2.upper()]

    dist_ranges = [(corner - 30, corner + 30) for corner in corners]
    query = f" SELECT driver_number, lap_number, actual_distance, speed FROM telemetry WHERE ((driver_number = {driver1_number} and lap_number = {lap1_number}) OR (driver_number = {driver2_number} and lap_number = {lap2_number})) AND ("
    subquery = ' OR '.join([f'(actual_distance > {corner[0]} AND actual_distance < {corner[1]})' for corner in dist_ranges])
    query += subquery + ')'  

    # print(query)

    df = pd.read_sql_query(query, engine)
    # print(len(df))
    # df['date'] = pd.to_datetime(df.date, format='ISO8601')
    df[['actual_distance', 'speed']] = df[['actual_distance', 'speed']].astype(float)

    # print(df)

    groups = df.groupby(['driver_number', 'lap_number'])
    df1 = groups.get_group((driver1_number, lap1_number)).sort_values(by = 'actual_distance')
    df2 = groups.get_group((driver2_number, lap2_number)).sort_values(by = 'actual_distance')

    dist1 = gaussian_filter1d(df1.actual_distance, sigma = 10)
    dist2 = gaussian_filter1d(df2.actual_distance, sigma = 10)

    data1 = [driver_config['driver_code'][driver1_number]] + [df1[(df1.actual_distance < corner[1]) & (df1.actual_distance > corner[0])].speed.round(0).min() for corner in dist_ranges]
    data2 = [driver_config['driver_code'][driver2_number]] + [df2[(df2.actual_distance < corner[1]) & (df2.actual_distance > corner[0])].speed.round(0).min() for corner in dist_ranges]
    # print(data1, data2)
    columns = ['driver_code'] + [str(x) for x in range(1, 1+ len(corners))]
    data = pd.concat([pd.DataFrame(data1).T, pd.DataFrame(data2).T])
    data.columns = columns
    return data.to_dict('records')

@app.callback(
    Output('laptime-plot', 'figure'),
    [Input('laptime-updater-component', 'n_intervals'),
     Input('laptime-threshold-input', 'value')]
)
def update_laptime_plot(n_intervals, laptime_threshold):
    # Replace this with your data update logic

    query = f"SELECT driver_number, lap_number, lap_duration FROM laptimes"
    '''TODO'''
    df = pd.read_sql_query(query, engine).dropna().astype(float).query(f"lap_duration < {laptime_threshold}")
    df['driver_order'] = df.driver_number.map(driver_config['driver_code']).map(driver_config['driver_order'])

    #df = get_data(f'https://api.openf1.org/v1/laps?session_key={session_key}')
    #df = df[['driver_number', 'lap_number', 'lap_duration']].dropna().astype(float).query(f"lap_duration < {laptime_threshold}")

    #df = pd.read_sql_query(query, engine).dropna().astype(float).query(f"lap_duration < {laptime_threshold}")


    # df_ = df.pivot(columns = 'lap_number', index = 'driver_number', values = 'lap_duration').round(3)
    # df_.columns = [int(x) for x in df_.columns]
    # df_ = df_[sorted(df_.columns.tolist())].reset_index()



    traces = []
    for k, v in df.sort_values(by = ['driver_order', 'lap_number']).groupby('driver_order'):
      v[['driver_number','lap_number']] = v[['driver_number','lap_number']].astype(int)
      v.set_index('lap_number', inplace=True)
      x = np.arange(1,max(v.index)+1)
      y = [v['lap_duration'].loc[i] if float(i) in v.index.values else None for i in range(1,max(v.index)+1) ]
      traces.append(go.Scatter(x=x, y=y, mode='markers+lines', name=f'{driver_config['driver_code'][v.driver_number.iloc[0]]}', line=lines[v.driver_number.iloc[0]]))
    layout = go.Layout(title = f'''Laptime Data''', xaxis=dict(title='Lap Number'), yaxis=dict(title='Time'), uirevision = 8)
    figure = go.Figure(data=traces, layout=layout)
    return figure

@app.callback(
    Output('maxspeed-table', 'columns'),
    [Input('maxspeed-updater-component', 'n_intervals')]
)
def update_maxspeed_columns(n_intervals):
    query = f"select max(lap_number) as max_lap_number from telemetry"
    df = pd.read_sql_query(query, engine)
    columns = ['driver_code'] + [str(lap) for lap in range(0, 1 + df.iloc[0].max_lap_number)] 
    columns = [{'name': col, 'id': col} for col in columns]
    # print(columns)
    return columns

@app.callback(
    Output('samples-table', 'columns'),
    [Input('maxspeed-updater-component', 'n_intervals')]
)
def update_samples_columns(n_intervals):
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
    df_['driver_code'] = df_.driver_number.map(driver_config['driver_code'])
    df_.columns = [str(x) for x in df_.columns]
    # print(df_.head(2))

    # traces = []
    # for k, v in df.groupby('driver_number'):
    #   traces.append(go.Scatter(x=v['lap_number'], y=v['lap_duration'], mode='markers+lines', name=f'{driver_config['driver_code'][int(k)]}'))
    # layout = go.Layout(title = f'''Laptime Data''', xaxis=dict(title='Lap Number'), yaxis=dict(title='Time'), uirevision = 8)
    # figure = go.Figure(data=traces, layout=layout)
    return df_.to_dict('records')

@app.callback(
    Output('samples-table', 'data'),
    [Input('maxspeed-updater-component', 'n_intervals')]
)
def update_samples_table(n_intervals):
    # Replace this with your data update logic

    query = f"select driver_number, lap_number, count(*) as sample_count from telemetry group by driver_number, lap_number"
    df = pd.read_sql_query(query, engine)
    df_ = df.pivot(columns = 'lap_number', index = 'driver_number', values = 'sample_count').round(0)
    df_.columns = [int(x) for x in df_.columns]
    df_ = df_[sorted(df_.columns.tolist())].reset_index()
    df_['driver_code'] = df_.driver_number.map(driver_config['driver_code'])
    df_.columns = [str(x) for x in df_.columns]
    # print(df_.head(2))

    # traces = []
    # for k, v in df.groupby('driver_number'):
    #   traces.append(go.Scatter(x=v['lap_number'], y=v['lap_duration'], mode='markers+lines', name=f'{driver_config['driver_code'][int(k)]}'))
    # layout = go.Layout(title = f'''Laptime Data''', xaxis=dict(title='Lap Number'), yaxis=dict(title='Time'), uirevision = 8)
    # figure = go.Figure(data=traces, layout=layout)
    return df_.to_dict('records')

@app.callback(
    Output('position-table', 'data'),
    [Input('position-updater-component', 'n_intervals')]
)
def update_position_table(n_intervals):
    # Replace this with your data update logic

    query = f"select * from position"
    df = pd.read_sql_query(query, engine)
    df[['driver_number', 'position']] = df[['driver_number', 'position']].astype(int)
    df['driver_code'] = df['driver_number'].map(driver_config['driver_code'])
    df['date'] = pd.to_datetime(df['date'], format='mixed').dt.strftime('%H:%M:%S')

    query = f"select driver_number, max(lap_number) as lap_number from telemetry group by driver_number"
    df_laps = pd.read_sql_query(query, engine)

    #query = 'select driver_number, min(lap_duration) as fastest_lap, lap_number as fast_lap_number from laptimes group by driver_number'
    #df_fast_laps = pd.read_sql_query(query, engine)
    #df_fast_laps['driver_number'] = df_fast_laps['driver_number'].astype(int)
    #df_fast_laps['fastest_lap'] = df_fast_laps['fastest_lap'].astype(float)

    #query = 'select driver_number, '

    df = df.merge(df_laps, on = 'driver_number') #.merge(df_fast_laps, on='driver_number')
    return df[['driver_code', 'position', 'date', 'lap_number']].sort_values(by = 'position').to_dict('records')

@app.callback(
    Output('track-location-plot', 'figure'),
    [Input('track-location-updater-component', 'n_intervals')]
)
def update_track_location_plot(n_intervals):

    # this query works with the assumption that all drivers get telemetry at the same timestamps (which is what we have observed)
    query = f"SELECT x, y, driver_number, date FROM telemetry where date = (select max(date) from telemetry) group by driver_number"
    df = pd.read_sql_query(query, engine)
    df['driver_order'] = df.driver_number.map(driver_config['driver_code']).map(driver_config['driver_order'])
    df['driver_code'] = df.driver_number.map(driver_config['driver_code'])
    
    df_layout = pd.read_csv(f'track_layout/{location}-{year}.csv')
    traces = []
    traces.append(go.Scatter(x=df_layout.x, y=df_layout.y, mode='lines', line=dict(dash='dot',color='#404040', width = 3), hoverinfo='skip', showlegend=False))
    
    for k, v in df.sort_values(by = ['driver_order']).groupby('driver_order'):
      traces.append(go.Scatter(x=v['x'], y=v['y'], mode='markers', marker={'size': 18, 'color': f'{driver_config['team_colour'][driver_config['driver_code'][v.driver_number.iloc[0]]]}'}, name=f'{driver_config['driver_code'][v.driver_number.iloc[0]]}', legendgroup=f'{driver_config['driver_code'][v.driver_number.iloc[0]]}'))
    
    annotations=[
        dict(
            x= xi + np.clip(500 * np.abs(xi)/(xi**2 + yi**2 + 1)**0.5, 200, 400) * np.sign(xi),
            y= yi + np.clip(500 * np.abs(yi)/(xi**2 + yi**2 + 1)**0.5, 200, 400) * np.sign(yi),
            text=text,
            showarrow=False,
            font=dict(
                family= 'Arial',
                size = 18,
                color= f'{driver_config['team_colour'][text]}',
                # weight='bold'
             ),  # Making the text bold
            # legendgroup = text,
            # xanchor='center',
            # yanchor='bottom',
        )
        for xi, yi, text in zip(df.x, df.y, df.driver_number.map(driver_config['driver_code']))
    ]
    layout = go.Layout(title = f'''Track Location {df.date.iloc[0]}''', xaxis=dict(title='X'), yaxis=dict(title='Y'), uirevision = 8, height=800, width=800, yaxis_range=[df_layout.y.min()-500,df_layout.y.max()+500], xaxis_range=[df_layout.x.min()-500,df_layout.x.max()+500], annotations = annotations)
    figure = go.Figure(data=traces, layout=layout)
    return figure
    
    # for k, v in df.sort_values(by = ['driver_order']).groupby('driver_order'):

    #   text = v.driver_code.iloc[0]
    #   xi = v.x.iloc[0]
    #   yi = v.y.iloc[0]
      
    #   traces.append(go.Scatter(x=[xi], y=[yi], mode='markers', marker={'size': 18, 'color': f'{driver_config['team_colour'][driver_config['driver_code'][v.driver_number.iloc[0]]]}'}, name=f'{driver_config['driver_code'][v.driver_number.iloc[0]]}', text = text, textposition='top left'))
    #       #                   text = dict(
    #       #     x= xi + np.clip(500 * np.abs(xi)/(xi**2 + yi**2 + 1)**0.5, 200, 400) * np.sign(xi),
    #       #     y= yi + np.clip(500 * np.abs(yi)/(xi**2 + yi**2 + 1)**0.5, 200, 400) * np.sign(yi),
    #       #     text=text,
    #       #     showarrow=False,
    #       #     font=dict(
    #       #         family= 'Arial',
    #       #         size = 18,
    #       #         color= f'{driver_config['team_colour'][text]}',
    #       #      ), 
    #       #     legendgroup = text,
    #       # )
      
    # layout = go.Layout(title = f'''Track Location {df.date.iloc[0]}''', xaxis=dict(title='X'), yaxis=dict(title='Y'), uirevision = 8, height=800, width=800, yaxis_range=[df_layout.y.min()-500,df_layout.y.max()+500], xaxis_range=[df_layout.x.min()-500,df_layout.x.max()+500])
    # figure = go.Figure(data=traces, layout=layout)
    # return figure
    
    
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
    layout = go.Layout(title = f'''Weather Data''', xaxis=dict(title='Time'), yaxis=dict(title='Value'), uirevision=8)
    figure = go.Figure(data=traces, layout=layout)
    return figure

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port = 8050, host = '0.0.0.0')
