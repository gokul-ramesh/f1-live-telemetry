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

import warnings
warnings.filterwarnings("ignore")

# Bahrain 
# corners = [711.9508171931816, 814.1876569292108, 936.7201493049793, 1506.9841172483643, 1787.9086361225473, 1880.658880160338, 1975.1619299965303, 2232.6789270606096, 2598.3923945375827, 2691.728150228979, 3466.886530179646, 3875.788302688632, 4084.7641184324057, 4889.970687598453, 4970.110027289251]
# corners = [ 354.42083043,  439.27690256, 1080.70076863, 1232.24836987,
#        1432.90920875, 1851.34924493, 1950.29260385, 2152.99296282,
#        3258.34533469, 3367.69508844, 4078.44514627, 4341.73730817,
#        4580.99734427, 4742.19833527] #Australia
# # Azerbaijan
# corners = [840.4395031422582, 925.2192705531841, 1168.2113212093552, 1741.685572018549, 2122.736856995752, 2364.579752964712, 2547.5431414033974, 2637.1095897460928, 2925.3814319017774, 3499.034858626688, 3638.214025589213, 3781.746397745988, 4025.8641801250633, 4156.471023673764, 4203.5096385535135, 4371.906121536575]

#Session and circuit information

track_config = pd.read_csv('config/track_config.csv')
pit_limit = 80

location = sys.argv[1]
year = int(sys.argv[2])
needed_session = sys.argv[3]

track = track_config.query(f''' circuit_location == '{location}' ''')

TOTAL_LAPS = int(track.total_laps.iloc[0])
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
team_groups = {k: v['name_acronym'].tolist() for k, v in driver_data[['team_acronym', 'name_acronym']].groupby('team_acronym')}

driver_config = {}
driver_config['driver_code'] = {}
driver_config['driver_number'] = {}
driver_config['team_name'] = {}
driver_config['team_colour'] = {}
driver_config['team_order'] = {}
driver_config['driver_order'] = {}
# driver_config['team_code'] = {}

for ind, driver in driver_data.sort_values(by = ['team_order', 'driver_number']).iterrows():
    driver_config['driver_number'][driver['name_acronym']] = driver['driver_number']
    driver_config['driver_code'][driver['driver_number']] = driver['name_acronym']
    driver_config['team_name'][driver['name_acronym']]= f'''{driver['team_name']}'''
    driver_config['team_colour'][driver['name_acronym']]= f'''#{driver['team_colour']}'''
    # driver_config['team_code'][driver['name_acronym']]= f'''{driver['team_acronym']}'''
    driver_config['team_order'][driver['name_acronym']] = driver['team_order']
    driver_config['driver_order'][driver['name_acronym']] = driver['driver_order']

drs_map = {k:50 if k in [10,12,14] else 0 for k in range(15)}

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

group1_buttons = html.Div(
    [
        dbc.Checklist(
            id="group1-buttons",
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
group2_buttons = html.Div(
    [
        dbc.Checklist(
            id="group2-buttons",
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
group3_buttons = html.Div(
    [
        dbc.Checklist(
            id="group3-buttons",
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

group_button_group = html.Div(
    [
        dbc.RadioItems(
            id="group-radiobuttons",
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-secondary",
            labelCheckedClassName="active",
                options=[{'label': html.Div([group], style={'color':'Black', 'font-size':16, 'text-align':'center'}), 'value': group} for group in list(team_groups.keys()) + ['G1', 'G2', 'G3', 'ALL']],
        value='ALL',
        ),
        html.Div(id='output'),
    ],
    className='radio-group',
)

fuel_toggle_group = html.Div(
    [
        dbc.RadioItems(
            id="fuel-toggle-radiobuttons",
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-secondary",
            labelCheckedClassName="active",
                options=[{'label': html.Div([group[0]], style={'color':'Black', 'font-size':16, 'text-align':'center'}), 'value': group[1]} for group in [('Regular', 0), ('Corrected', 1)]],
        value=0,
        ),
        html.Div(id='output'),
    ],
    className='radio-group',
)

# df_position = pd.DataFrame(columns = ['driver_code', 'position', 'date', 'lap_number', 'fastest', 'f_lap','n', 'n-1', 'n-2'])

# Define the layout of your app
app.layout = html.Div([

    html.H1(f"Laptime Comparison for Race {location}, {year}"),
    dbc.Col([
            dbc.Row([driver1_button_group]),
            dbc.Row([lap1_button_group]),
            dbc.Row([driver2_button_group]),
            dbc.Row([lap2_button_group]),
        ], align = 'left',
    ),
    html.Button('Submit', id='telemetry-submit-value', n_clicks=0),
    dcc.Interval(id='telemetry-updater-component', interval=5000, n_intervals=0),
    dcc.Interval(id='weather-updater-component', interval=30000, n_intervals=0),
    dcc.Interval(id='laptime-updater-component', interval=10000, n_intervals=0),
    dcc.Interval(id='race-control-updater-component', interval=10000, n_intervals=0),
    dcc.Interval(id='maxspeed-updater-component', interval=15000, n_intervals=0),
    dcc.Interval(id='position-updater-component', interval=5000, n_intervals=0),
    dcc.Interval(id='track-location-updater-component', interval=5000, n_intervals=0),
    dcc.Interval(id='pitstop-updater-component', interval=10000, n_intervals=0),
    dcc.Interval(id='pitstop-formatting-updater-component', interval=2000, n_intervals=0),
    
    dcc.Graph(id='scatter-plot'),
    dash_table.DataTable(
        id='corner-minspeed-table',
        style_data={ 'border': '1px solid black' },
        style_header={ 'border': '2px solid black', 'textAlign' : 'center'},
        columns=[{'name': str(col), 'id': str(col)} for col in ['driver_code'] + [*range(1, 1+len(corners))]],
        fill_width=False,
    ),

    dbc.Col([
            dbc.Row([html.H5("Group 1"), group1_buttons]),
            dbc.Row([html.H5("Group 2"), group2_buttons]),
            dbc.Row([html.H5("Group 3"), group3_buttons]),
        ], align = 'left',
    ),
    dcc.Input(id="laptime-threshold-input", type="number", placeholder="", size = '5px', step=1, value = 200,),
    dbc.Col([dbc.Row([fuel_toggle_group])], align = 'left',),
    dbc.Col([dbc.Row([group_button_group]),], align = 'left',),
    dcc.Graph(id='laptime-plot'),
    
    html.Div([
        html.Div([
            html.H2("Positions"),
            dash_table.DataTable(
                id='position-table',
                style_data={'border': '1px solid black',},
                style_header={ 'border': '2px solid black', 'textAlign' : 'center', 'backgroundColor' : '#aaaaaa',},
                style_cell_conditional=[{'if': {'column_id': 'driver_code'}, 'textAlign': 'center',},   
                                        {'if': {'column_id': 'position'}, 'textAlign': 'center',},
                                        {'if': {'column_id': 'lap_number'}, 'textAlign': 'center',},
                                        {'if': {'column_id': 'fastest'}, 'textAlign': 'center',},
                                        {'if': {'column_id': 'f_lap'}, 'textAlign': 'center',},
                                        ],
                columns=[{'name': col, 'id': col} for col in ['date', 'driver_code', 'position',  'lap_number', 'fastest', 'f_lap', 'n', 'n-1', 'n-2']],
                fill_width=False,
            ),], 
            style={'width': '49%', 'display': 'inline-block'},
        ),
        html.Div([dcc.Graph(id='track-location-plot')], style={'width': '49%', 'display': 'inline-block'}),
        ], style={'display': 'flex'},
    ),
    
        html.Div([
        html.Div([
            html.H2("Pitstops"),
            dash_table.DataTable(
                id='pitstop-table',
                style_data={ 'border': '1px solid black' },
                style_header={ 'border': '2px solid black', 'textAlign' : 'center', 'backgroundColor' : '#aaaaaa'},
                style_cell_conditional=[{'if': {'column_id': 'driver_code'}, 'textAlign': 'center'}] +
                    [{'if': {'column_id': f'lap{i}'}, 'textAlign': 'center', 'backgroundColor' : '#cccccc'} for i in range(5)] + 
                    [{'if': {'column_id': f'pit{i}'}, 'textAlign': 'right',} for i in range(5)] +
                    [{'if': {'column_id': f'rest{i}'}, 'textAlign': 'right',} for i in range(5)],
                columns=[{'name': str(col), 'id': str(col)} for col in ['driver_code', 'out_lap_number', 'duration_pit', 'duration_rest']],
                fill_width=False,
            ),
        ], style={'width': '49%', 'display': 'inline-block'}),
        html.Div([
            html.H2("Race Control"),
            dash_table.DataTable(
                id='race-control-table',
                # filter_action="native",
                style_data={ 'border': '1px solid black' },
                style_header={ 'border': '2px solid black', 'textAlign' : 'center', 'backgroundColor' : '#aaaaaa'},
                style_cell={'whiteSpace': 'normal'},
                style_cell_conditional=[{'if': {'column_id': 'lap_number'}, 'textAlign': 'center'},
                                        {'if': {'column_id': 'flag'}, 'textAlign': 'center'},
                                        {'if': {'column_id': 'message'}, 'textAlign': 'left'}, ],
                style_table={'maxHeight': '600px', 'maxWidth': '900px', 'overflowY': 'scroll'},
                columns=[{'name': col, 'id': col} for col in ['date', 'lap_number', 'flag', 'message']],
                fill_width=False,
            ),
        ], style={'width': '49%', 'display': 'inline-block'}),
    ]),

    html.H2("Max Speed"),
    dash_table.DataTable(
        id='maxspeed-table',
        style_data={ 'border': '1px solid black'},
        style_header={ 'border': '2px solid black', 'textAlign' : 'center', 'backgroundColor' : '#aaaaaa'},
        style_cell_conditional=[{'if': {'column_id': 'driver_code'}, 'textAlign': 'center'}],
        columns=[{'name': col, 'id': col} for col in columns],
        fill_width=False,
    ),
  
    dcc.Graph(id='weather-plot'),
    html.H2("Samples"),
    dash_table.DataTable(
            id='samples-table',
            style_data={ 'border': '1px solid black' },
            style_header={ 'border': '2px solid black', 'textAlign' : 'center'},
            columns=[{'name': col, 'id': col} for col in columns],
            fill_width=False,
    ),
    
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
    df['drs'] = df['drs'].map(drs_map)

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
              go.Scatter(x=df2.actual_distance_smoothed, y=df2['speed'], mode='lines', name=f'{driver2.upper()}, Lap {lap2_number}', line=line_driver2, legendgroup='group2')]

    drss = [go.Scatter(x=df1.actual_distance_smoothed, y=df1['drs'], mode='lines', name=f'{driver1.upper()}', line=dict(color=f"{driver_config['team_colour'][driver1.upper()]}"), legendgroup='group1', showlegend=False),
              go.Scatter(x=df2.actual_distance_smoothed, y=df2['drs'], mode='lines', name=f'{driver2.upper()}', line=line_driver2, legendgroup='group2', showlegend=False)]

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
    fig = make_subplots(rows=6, cols=1, vertical_spacing = 0.005,
                        row_width = [0.12, 0.12, 0.12, 0.12, 0.22, 0.30]) # don't ask me how this works, the reverse of this row_width list is what I expected to work - it made the last plot the tallest
    
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
    for trace in drss:
        fig.add_trace(trace, row=1, col=1)

    plot_list = ["Speed/DRS", f"Live Delta<br><-- {driver2.upper()} faster | {driver1.upper()} faster -->", "Throttle", "Brake", "RPM", "Gear"]
    for i in range(6):
      if i == 0:
        fig['layout']['yaxis']['title']= plot_list[i]
        fig['layout']['xaxis']['range']= [0, circuit_length]
      else:
        fig['layout'][f'yaxis{i+1}']['title']= plot_list[i]
        fig['layout'][f'xaxis{i+1}']['range']= [0, circuit_length]
       
    # fig['layout']['yaxis']['title']="Speed/DRS"
    # fig['layout']['yaxis2']['title']="Delta"
    # fig['layout']['yaxis3']['title']="Throttle"
    # fig['layout']['yaxis4']['title']="Brake"
    # fig['layout']['yaxis5']['title']="RPM"
    # fig['layout']['yaxis6']['title']="Gear"
    fig.update_layout(uirevision=8, height=1400, width=1800, title_text=f'''{driver1.upper()} : {get_lap_dur(driver1_number, lap1_number)}s, {driver2.upper()}: {get_lap_dur(driver2_number,lap2_number)}s''')
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(minor=dict(tickvals=np.arange(0,350,10), tickmode='array', showgrid=True, ticks="inside"))

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
    [
     Input('group-radiobuttons', 'value'),
     Input('laptime-threshold-input', 'value'),
     Input('fuel-toggle-radiobuttons', 'value'),
     Input('group1-buttons', 'value'),
     Input('group2-buttons', 'value'),
     Input('group3-buttons', 'value'),
     Input('laptime-updater-component', 'n_intervals'),
     ]
)
def update_laptime_plot(group_name, laptime_threshold, is_corrected, group1, group2, group3, n_intervals):
    # Replace this with your data update logic

    def fuel_corrected_laptime(lap_duration, lap_number, TOTAL_LAPS, TOTAL_FUEL = 110, TIME_LOST_PER_KG = 0.035):
        return lap_duration - (1 - lap_number/TOTAL_LAPS) * TOTAL_FUEL * TIME_LOST_PER_KG   
    
    team_groups.update({'G1':group1})
    team_groups.update({'G2':group2})
    team_groups.update({'G3':group3})
    team_groups['ALL'] = driver_config['driver_number'].keys()
    query = f"SELECT driver_number, lap_number, lap_duration FROM laptimes"
    df = pd.read_sql_query(query, engine).dropna().astype(float).query(f"lap_duration < {laptime_threshold}")
    df['lap_duration_fuel_corrected'] = df.apply(lambda x: fuel_corrected_laptime(x.lap_duration, x.lap_number, TOTAL_LAPS), axis = 1)
    df['driver_order'] = df.driver_number.map(driver_config['driver_code']).map(driver_config['driver_order'])

    #df = get_data(f'https://api.openf1.org/v1/laps?session_key={session_key}')
    #df = df[['driver_number', 'lap_number', 'lap_duration']].dropna().astype(float).query(f"lap_duration < {laptime_threshold}")

    #df = pd.read_sql_query(query, engine).dropna().astype(float).query(f"lap_duration < {laptime_threshold}")


    # df_ = df.pivot(columns = 'lap_number', index = 'driver_number', values = 'lap_duration').round(3)
    # df_.columns = [int(x) for x in df_.columns]
    # df_ = df_[sorted(df_.columns.tolist())].reset_index()


    traces = []
    visibility = {driver_no: True if driver_code in team_groups[group_name] else "legendonly" for driver_code, driver_no in driver_config['driver_number'].items()}
    for k, v in df.sort_values(by = ['driver_order', 'lap_number']).groupby('driver_order'):
      v[['driver_number','lap_number']] = v[['driver_number','lap_number']].astype(int)
      v.set_index('lap_number', inplace=True)
      x = np.arange(1,max(v.index)+1)
      y = [v['lap_duration_fuel_corrected' if is_corrected == 1 else 'lap_duration'].loc[i] if float(i) in v.index.values else None for i in range(1,max(v.index)+1) ]
      traces.append(go.Scatter(x=x, y=y, mode='markers+lines', name=f'{driver_config["driver_code"][v.driver_number.iloc[0]]}', line=lines[v.driver_number.iloc[0]], visible=visibility[v.driver_number.iloc[0]]))
    layout = go.Layout(title = 'Fuel Corrected Laptimes' if is_corrected == 1 else 'Regular Laptimes', xaxis=dict(title='Lap Number'), yaxis=dict(title='Time'), uirevision = 8, modebar_add=["v1hovermode","toggleSpikelines",])
    figure = go.Figure(data=traces, layout=layout)

    #figure.update_traces(visible='legendonly', selector=dict(name=team_groups[groups][0]))
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
    df = pd.read_sql_query(query, engine).astype(int)
    df_ = df.pivot(columns = 'lap_number', index = 'driver_number', values = 'max_speed').round(0)
    df_.columns = [int(x) for x in df_.columns]
    df_ = df_[sorted(df_.columns.tolist())].reset_index()
    df_['driver_code'] = df_.driver_number.map(driver_config['driver_code'])
    df_.columns = [str(x) for x in df_.columns]
    df_['driver_order'] = df_.driver_code.map(driver_config['driver_order'])
    # print(df_.head(2))

    # traces = []
    # for k, v in df.groupby('driver_number'):
    #   traces.append(go.Scatter(x=v['lap_number'], y=v['lap_duration'], mode='markers+lines', name=f'{driver_config['driver_code'][int(k)]}'))
    # layout = go.Layout(title = f'''Laptime Data''', xaxis=dict(title='Lap Number'), yaxis=dict(title='Time'), uirevision = 8)
    # figure = go.Figure(data=traces, layout=layout)
    return df_.sort_values(by = 'driver_order').to_dict('records')

@app.callback(
    Output('race-control-table', 'data'),
    [Input('race-control-updater-component', 'n_intervals')]
)
def update_race_control_table(n_intervals):
    # Replace this with your data update logic

    query = f"select * from race_control"
    df = pd.read_sql_query(query, engine)[['date', 'lap_number', 'flag', 'message']]
    df['date'] = pd.to_datetime(df['date'], format='mixed').dt.time
    return df.sort_values(by = 'date', ascending = False).to_dict('records')

@app.callback(
    Output('race-control-table', 'style_data_conditional'),
    [Input('race-control-table','data'),
     Input('race-control-updater-component', 'n_intervals')]
)
def update_race_control_style_data_conditional(data, n_intervals):

    mapping = {'GREEN':'green', 'YELLOW':'yellow', 'DOUBLE YELLOW':"orange", 'CLEAR':'green', 'BLUE':'cyan', 'CHEQUERED':'light gray'}
    df = pd.DataFrame(data).fillna(0)
    conditional_styles = []
    for i in range(len(df)):
        flag = df.flag.iloc[i]
        if flag in mapping.keys():
            conditional_styles.append({'if': {'row_index': i}, 'backgroundColor': mapping[flag], 'color': 'black'})

    return conditional_styles

@app.callback(
    Output('maxspeed-table', 'style_data_conditional'),
    [Input('maxspeed-table','data'),
     Input('maxspeed-updater-component', 'n_intervals')]
)
def update_maxspeed_style_data_conditional(data, n_intervals):

    def get_color(x):
        if x is None:
            return 0
        else:
            return np.clip((64 + 172 * (int(x) - 250)/100), 0, 224)

    df = pd.DataFrame(data).fillna(0)

    query = f"select driver_number, lap_number, max(drs) as max_drs from telemetry group by driver_number, lap_number"
    df_ = pd.read_sql_query(query, engine).astype(int)
    df_ = df_.pivot(columns = 'lap_number', index = 'driver_number', values = 'max_drs').round(0).fillna(0) >= 10
    df_ = df_.reset_index()
    

    df_['driver_code'] = df_.driver_number.map(driver_config['driver_code'])
    df_['driver_order'] = df_.driver_code.map(driver_config['driver_order'])

    df['driver_code'] = df.driver_number.map(driver_config['driver_code'])
    df['driver_order'] = df.driver_code.map(driver_config['driver_order'])

    df_ = df_.sort_values(by = 'driver_order')
    df = df.sort_values(by = 'driver_order')
    conditional_styles = []

    for i in range(len(data)):
        conditional_styles.append({'if': {'column_id': 'driver_code', 'row_index': i}, 'backgroundColor': driver_config['team_colour'][df.driver_code[i]], 'color': 'white'})
        conditional_styles += [{'if': {'column_id': str(column), 'row_index': i}, 'border': '3px solid yellow', 'color' : 'white', 'backgroundColor': f'rgb({get_color(df[str(column)].iloc[i])}, 0, 0, 1)' }if df_[column].iloc[i] == 1 else {'if': {'column_id': str(column), 'row_index': i}, 'color' : 'white', 'backgroundColor': f'rgb({get_color(df[str(column)].iloc[i])}, 0, 0, 1'} for column in df_.columns[1:-2]]
    # print(int(x) for x in df.columns[:-2])
    # conditional_styles += [{'if': {'filter_query': '{%s} > 330' % column, 'column_id': column}, 'border': '5px solid red'}  for column in df.columns[:-2]]

    return conditional_styles


@app.callback(
    Output('pitstop-table', 'columns'),
    [Input('pitstop-updater-component', 'n_intervals')]
)
def update_pitstop_columns(n_intervals):

    query = f"select * from laptimes"
    df_laps = pd.read_sql_query(query, engine).query("is_pit_out_lap == 'True'")[['driver_number', 'lap_number']].astype(int)
    merged_windows = pd.DataFrame({
    'pit_stops': df_laps.groupby(['driver_number'])['lap_number'].agg(lambda x: len(x.unique())),
    })
    max_pits = 1 if len(merged_windows) == 0 else max(merged_windows.pit_stops) 
    columns = ['driver_code'] + [item for sublist in [[f'lap{i+1}', f'pit{i+1}', f'rest{i+1}'] for i in range(max_pits)] for item in sublist]
    columns = [{'name': col, 'id': col} for col in columns]
    return columns

@app.callback(
    Output('position-table', 'style_data_conditional'),
    [Input('position-table','data'),
     Input('pitstop-formatting-updater-component', 'n_intervals')]
)
def update_position_style_data_conditional(data, n_intervals):

    def get_color(x):
        # return np.clip(127 + int(128 * x/30), 127, 255)
        return np.clip((255 - 128 * x/30), 0, 255)
    
    def get_color_2(x):
        return np.clip((255 - 32 * x), 128, 255)

    df = pd.DataFrame(data)
    fastest_lap = df.fastest.min()
    current_lap = df.lap_number.max()
    conditional_styles = []

    for i in range(len(data)):
        if df['n'][i] > df['fastest'][i]:
            conditional_styles.append({'if': {'column_id': 'n', 'row_index': i}, 'backgroundColor': '#ffcc00',  'color': 'black',})
        else:
            conditional_styles.append({'if': {'column_id': 'n', 'row_index': i}, 'backgroundColor': '#00cc44',  'color': 'black',})

        if df['n-1'][i] > df['fastest'][i]:
            conditional_styles.append({'if': {'column_id': 'n-1', 'row_index': i}, 'backgroundColor': '#ffcc00',  'color': 'black',})
        else:
            conditional_styles.append({'if': {'column_id': 'n-1', 'row_index': i}, 'backgroundColor': '#00cc44',  'color': 'black',})

        if df['n-2'][i] > df['fastest'][i]:
            conditional_styles.append({'if': {'column_id': 'n-2', 'row_index': i}, 'backgroundColor': '#ffcc00',  'color': 'black',})
        else:
            conditional_styles.append({'if': {'column_id': 'n-2', 'row_index': i}, 'backgroundColor': '#00cc44',  'color': 'black',})

        color = get_color(df['lap_number'][i] - df['f_lap'][i])
        color_2 = get_color_2(current_lap - df['lap_number'][i])
        conditional_styles.append({'if': {'column_id': 'f_lap', 'row_index': i}, 'backgroundColor': f'''rgba({color}, {color}, {color}, 1)''', 'color': 'black'})
        conditional_styles.append({'if': {'column_id': 'driver_code', 'row_index': i}, 'backgroundColor': driver_config['team_colour'][df.driver_code[i]], 'color': 'white'})
        conditional_styles.append({'if': {'column_id': 'date', 'row_index': i}, 'backgroundColor': driver_config['team_colour'][df.driver_code[i]], 'color': 'white'})
        conditional_styles.append({'if': {'column_id': 'lap_number', 'row_index': i}, 'backgroundColor': f'''rgba({color_2}, {color_2}, {color_2}, 1)''', 'color': 'black'})
        conditional_styles.append({'if': {'column_id': 'position', 'row_index': i}, 'backgroundColor': driver_config['team_colour'][df.driver_code[i]], 'color': 'white'})


    
        # style_data_conditional = [{'if': {'column_id': x, 'row_index': y},
        #              'backgroundColor': '#3D9970','color': 'white'} for x,y  in zip((df.idxmax(axis=1)), df.index)]


        # if df['n-1'][i] > df['fastest'][i]:
        #     conditional_styles.append({'if': {'row_index': i}, 'backgroundColor': '#ffcc00',  'color': 'white',})
        # else:
        #     conditional_styles.append({'if': {'row_index': i}, 'backgroundColor': '#00cc44',  'color': 'white',})

        # if df['n-2'][i] > df['fastest'][i]:
        #     conditional_styles.append({'if': {'row_index': i}, 'backgroundColor': '#ffcc00',  'color': 'white',})
        # else:
        #     conditional_styles.append({'if': {'row_index': i}, 'backgroundColor': '#00cc44',  'color': 'white',})
            
    conditional_styles.append({'if': { 'filter_query': '{{fastest}} = {}'.format(fastest_lap), 'column_id': 'fastest'}, 'backgroundColor': '#ff66cc', 'color': 'black'})
    conditional_styles.append({'if': { 'filter_query': '{{n}} = {}'.format(fastest_lap), 'column_id': 'n'}, 'backgroundColor': '#ff66cc', 'color': 'black',})
    conditional_styles.append({'if': { 'filter_query': '{{n-1}} = {}'.format(fastest_lap), 'column_id': 'n-1'}, 'backgroundColor': '#ff66cc', 'color': 'black',})
    conditional_styles.append({'if': { 'filter_query': '{{n-2}} = {}'.format(fastest_lap), 'column_id': 'n-2'}, 'backgroundColor': '#ff66cc', 'color': 'black',})
    
    return conditional_styles

    
    # df = pd.DataFrame(data)
    # fastest_lap = df.fastest.min()

    # return [{
    #             'if' : {'filter_query': '{{fastest}} = {}'.format(fastest_lap), 'column_id': 'fastest'},
    #             'backgroundColor': '#ff66cc', 'color': 'white'
    #         }] + [{
    #             'if': { 'filter_query': '{{n}} > {{fastest}}'},
    #             'backgroundColor': '#ffcc00', 'color': 'black',
    #         }]
    #         #   + [{
    #         #     'if': { 'filter_query': '{{n}} <= {{fastest}}', 'column_id': 'n'},
    #         #     'backgroundColor': '#00cc44', 'color': 'black',
    #         # }]  
    #         # + [{
    #         #     'if': { 'filter_query': '{{n-1}} > {}'.format(fastest), 'column_id': 'n-1'},
    #         #     'backgroundColor': '#ffcc00', 'color': 'black',
    #         # } for fastest in df['fastest']
    #         # ] + [{
    #         #     'if': { 'filter_query': '{{n-1}} <= {}'.format(fastest), 'column_id': 'n-1'},
    #         #     'backgroundColor': '#00cc44', 'color': 'black',
    #         # } for fastest in df['fastest']
    #         # ]  + [{
    #         #     'if': { 'filter_query': '{{n-2}} > {}'.format(fastest), 'column_id': 'n-2'},
    #         #     'backgroundColor': '#ffcc00', 'color': 'black',
    #         # } for fastest in df['fastest']
    #         # ] + [{
    #         #     'if': { 'filter_query': '{{n-2}} <= {}'.format(fastest), 'column_id': 'n-2'},
    #         #     'backgroundColor': '#00cc44', 'color': 'black',
    #         # } for fastest in df['fastest']
    #         # ] + [{
    #         #     'if': { 'filter_query': '{{n}} = {}'.format(fastest_lap), 'column_id': 'n'},
    #         #     'backgroundColor': '#ff66cc', 'color': 'white',  
    #         # }] + [{
    #         #     'if': { 'filter_query': '{{n-1}} = {}'.format(fastest_lap), 'column_id': 'n-1'},
    #         #     'backgroundColor': '#ff66cc', 'color': 'white',  
    #         # }] + [{
    #         #     'if': { 'filter_query': '{{n-2}} = {}'.format(fastest_lap), 'column_id': 'n-2'},
    #         #     'backgroundColor': '#ff66cc', 'color': 'white',  
    #         # }]
    #         # }] + [{
    #         #     'if': { 'filter_query': '{{n}} > {}'.format(fastest), 'column_id': 'n'},
    #         #     'backgroundColor': '#ffcc00', 'color': 'black',
    #         # } for fastest in df['fastest']
    #         # ] + [{
    #         #     'if': { 'filter_query': '{{n}} < {}'.format(fastest), 'column_id': 'n'},
    #         #     'backgroundColor': '#00cc44', 'color': 'black',
    #         # } for fastest in df['fastest']
    #         # ]             

@app.callback(
    Output('pitstop-table', 'style_data_conditional'),
    [Input('pitstop-table','data'),
     Input('pitstop-updater-component', 'n_intervals')]
)
def update_pitstop_style_data_conditional(data, n_intervals):

    # def get_color(x):
    #     # return np.clip(127 + int(128 * x/30), 127, 255)
    #     return np.clip((255 - 128 * x/30), 0, 255)

    df = pd.DataFrame(data)
    conditional_styles = []

    for i in range(len(data)):
        conditional_styles.append({'if': {'column_id': 'driver_code', 'row_index': i}, 'backgroundColor': driver_config['team_colour'][df.driver_code[i]], 'color': 'white'})
            
    # conditional_styles.append({'if': { 'filter_query': '{{fastest}} = {}'.format(fastest_lap), 'column_id': 'fastest'}, 'backgroundColor': '#ff66cc', 'color': 'black'})
    # conditional_styles.append({'if': { 'filter_query': '{{n}} = {}'.format(fastest_lap), 'column_id': 'n'}, 'backgroundColor': '#ff66cc', 'color': 'black',})
    # conditional_styles.append({'if': { 'filter_query': '{{n-1}} = {}'.format(fastest_lap), 'column_id': 'n-1'}, 'backgroundColor': '#ff66cc', 'color': 'black',})
    # conditional_styles.append({'if': { 'filter_query': '{{n-2}} = {}'.format(fastest_lap), 'column_id': 'n-2'}, 'backgroundColor': '#ff66cc', 'color': 'black',})
    
    return conditional_styles
    # return None

@app.callback(
    Output('pitstop-table', 'data'),
    [Input('pitstop-updater-component', 'n_intervals')]
)
def update_pitstop_table(n_intervals):
    # Replace this with your data update logic

    query = f"select * from laptimes"
    df_laps = pd.read_sql_query(query, engine).query("is_pit_out_lap == 'True'")[['driver_number', 'lap_number']].astype(int)
    df_laps = df_laps.sort_values(by = ['driver_number', 'lap_number'])
    pit_out_laps = {(v.driver_number, v.lap_number): int(v.lap_number) for k, v in df_laps.iterrows()}
    pit_laps = {(v.driver_number, v.lap_number - 1): int(v.lap_number) for k, v in df_laps.iterrows()}
    pit_laps.update(pit_out_laps)

    if len(pit_out_laps) == 0:
      return pd.DataFrame(columns = ['driver_code', 'out_lap_number', 'duration_pit', 'duration_rest']).to_dict('records')
    
    query = f"select * from telemetry where ("
    # subquery1 = ''.join([f'''(driver_number = '{k[0]}' and lap_number = '{k[1]}') or ''' for k in pit_laps.keys()] )
    subquery1 = ''.join([f'''(driver_number = '{k[0]}' and lap_number in ({k[1]}, {k[1]-1})) or ''' for k in pit_out_laps.keys()])
    query = query+ subquery1[:-3] + ')'
    df = pd.read_sql_query(query, engine)
    
    # df_pit = df[((df.actual_distance < 500) | (df.actual_distance > circuit_length - 500)) & (df.speed < pit_limit + 1)]
    df_pit = df[(abs(df.distance_l2) < 500) & (df.speed < pit_limit + 1)] #works on monaco, other places?
    df_pit['date'] = pd.to_datetime(df_pit['date'], format = 'mixed')
    df_pit = df_pit.groupby(['driver_number', 'lap_number']).agg(duration = ('date', np.ptp), start_date = ('date', 'min'), end_date = ('date', 'max')).reset_index()
    df_pit['duration'] = df_pit['duration'].dt.total_seconds()
    df_pit['pit_lap_number'] = df_pit.apply(lambda x: pit_laps[(x.driver_number, x.lap_number)], axis = 1)
    df_pit = df_pit[['driver_number', 'pit_lap_number', 'duration']].groupby(['driver_number', 'pit_lap_number']).agg('sum').reset_index().rename(columns = {'pit_lap_number':'lap_number'}).round(3)
    # print('df_pit len', len(df_pit))

    # df_rest = df[((df.actual_distance < 500) | (df.actual_distance > circuit_length - 500)) & (df.speed < 1)]
    df_rest = df[(abs(df.distance_l2) < 500) & (df.speed < 1)] #works on monaco, other places?

    df_rest['date'] = pd.to_datetime(df_rest['date'], format = 'mixed')
    df_rest = df_rest.groupby(['driver_number', 'lap_number']).agg(duration = ('date', np.ptp)).reset_index()
    df_rest['duration'] = df_rest['duration'].dt.total_seconds()
    # print('df_rest len', len(df_rest))

    data = []

    for k,v in df_rest.iterrows():
      # print((v.driver_number, v.lap_number))
      if (v.driver_number, v.lap_number) in pit_laps.keys():
          v['lap_number'] = int(pit_laps[(v.driver_number, v.lap_number)])
          data.append(v.T)
    # print(pit_laps.keys())
    # print('data len', len(data))
    df_rest = pd.concat(data, axis = 1).T
    
    df_merged = df_pit.merge(df_rest, on = ['driver_number', 'lap_number'], suffixes = ('_pit', '_rest'), how = 'inner').sort_values(by = ['driver_number', 'lap_number'])
    df_merged.lap_number = df_merged.lap_number.astype(int)
    df_merged.duration_pit = df_merged.duration_pit.round(2)
    df_merged.duration_rest = df_merged.duration_rest.round(2)
    # print(df_merged)
    
    # merged_windows = pd.DataFrame({
    #     'out_lap_number': df_merged.groupby(['driver_number'])['lap_number'].agg(lambda x: x.unique().tolist()),
    #     'duration_pit': df_merged.groupby(['driver_number'])['duration_pit'].agg(lambda x: x.unique().tolist()),
    #     'duration_rest': df_merged.groupby(['driver_number'])['duration_rest'].agg(lambda x: x.unique().tolist()),
    #     }).reset_index()
    # merged_windows['driver_code'] = merged_windows.driver_number.map(driver_config['driver_code'])
    # # print(merged_windows)

    # return merged_windows.astype(str).to_dict('records')

    merged_windows = pd.DataFrame({
    'lap_number': df_merged.groupby(['driver_number'])['lap_number'].agg(lambda x: (x.unique().astype(int).tolist())),
    'pit_stops': df_merged.groupby(['driver_number'])['lap_number'].agg(lambda x: len(x.unique())),
    'duration_pit': df_merged.groupby(['driver_number'])['duration_pit'].agg(lambda x: x.unique().round(2).tolist()),
    'duration_rest': df_merged.groupby(['driver_number'])['duration_rest'].agg(lambda x: x.unique().round(2).tolist()),
    }).reset_index()

    merged_windows['driver_code'] = merged_windows.driver_number.map(driver_config['driver_code'])

    # print(merged_windows)

    max_pits = max(merged_windows['pit_stops'])
    drivers = merged_windows['driver_code']
    laps = pd.DataFrame(merged_windows['lap_number'].to_list(), columns=[f'lap{i+1}' for i in range(max_pits)]).fillna('')
    pit_duration = pd.DataFrame(merged_windows['duration_pit'].to_list(), columns=[f'pit{i+1}' for i in range(max_pits)]).fillna('')
    rest_duration = pd.DataFrame(merged_windows['duration_rest'].to_list(), columns=[f'rest{i+1}' for i in range(max_pits)]).fillna('')
    merged_windows = pd.concat([drivers, laps, pit_duration, rest_duration], axis = 1)[['driver_code'] + [item for sublist in [[f'lap{i+1}', f'pit{i+1}', f'rest{i+1}'] for i in range(max_pits)] for item in sublist]]
    # return merged_windows.sort_values(by = [f'lap{i+1}' for i in range(max_pits)]).astype(str).to_dict('records')
    return merged_windows.sort_values(by = [f'lap{i+1}' for i in range(max_pits)]).to_dict('records')

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
    df_['driver_order'] = df_.driver_code.map(driver_config['driver_order'])
    df_.columns = [str(x) for x in df_.columns]
    # print(df_.head(2))

    # traces = []
    # for k, v in df.groupby('driver_number'):
    #   traces.append(go.Scatter(x=v['lap_number'], y=v['lap_duration'], mode='markers+lines', name=f'{driver_config['driver_code'][int(k)]}'))
    # layout = go.Layout(title = f'''Laptime Data''', xaxis=dict(title='Lap Number'), yaxis=dict(title='Time'), uirevision = 8)
    # figure = go.Figure(data=traces, layout=layout)
    return df_.sort_values(by = 'driver_order').to_dict('records')

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

    query = f"select driver_number, lap_duration, lap_number from laptimes"
    df_laptimes = pd.read_sql_query(query, engine).astype(float).dropna()
    df_laptimes[['driver_number','lap_number']]=df_laptimes[['driver_number','lap_number']].astype(int)
    df_fast = pd.DataFrame()
    df_fast[['driver_number','fastest', 'f_lap']] = df_laptimes.loc[df_laptimes.groupby('driver_number').lap_duration.idxmin()][['driver_number','lap_duration','lap_number']]
    l = pd.DataFrame()
    for k, v in df_laptimes.sort_values(by='lap_number').groupby('driver_number'):
        l = pd.concat([l, v[v.lap_number > max(v.lap_number) - 3].reset_index(drop=True).drop(columns=['lap_number'])])
    l.reset_index(inplace=True)
    latest = l.pivot(columns='index', values='lap_duration', index='driver_number').reset_index()
    latest = latest.reindex(columns=latest.columns.to_list()+['Latest', 'LatestBut1', 'LatestBut2'], fill_value=np.nan)
    if len(latest.columns) < 7:
        if 1 in latest.columns:
            latest[['Latest','LatestBut1']] = latest[[1,0]]
        elif 0 in latest.columns:
            latest['Latest'] = latest[0]
    else:
        latest[['Latest','LatestBut1','LatestBut2']] = latest.iloc[:,3:0:-1]

    df = df.merge(df_laps, on = 'driver_number').merge(df_fast, on='driver_number', how = 'outer').merge(latest, on='driver_number', how = 'outer').rename(columns = ({'Latest': 'n', 'LatestBut1':'n-1', 'LatestBut2':'n-2'}))

    return df[['date', 'driver_code', 'position', 'lap_number', 'fastest', 'f_lap', 'n', 'n-1', 'n-2']].sort_values(by = 'position').to_dict('records')

@app.callback(
    Output('track-location-plot', 'figure'),
    [
        Input('group-radiobuttons', 'value'),
        Input('group1-buttons', 'value'),
        Input('group2-buttons', 'value'),
        Input('group3-buttons', 'value'),
        Input('track-location-updater-component', 'n_intervals')]
)
def update_track_location_plot(group_name, group1, group2, group3, groupn_intervals):
    team_groups.update({'G1': group1})
    team_groups.update({'G2': group2})
    team_groups.update({'G3': group3})
    team_groups['ALL'] = driver_config['driver_number'].keys()

    # this query works with the assumption that all drivers get telemetry at the same timestamps (which is what we have observed)
    query = f"SELECT x, y, driver_number, date FROM telemetry where date = (select max(date) from telemetry) group by driver_number"
    df = pd.read_sql_query(query, engine)
    df['driver_order'] = df.driver_number.map(driver_config['driver_code']).map(driver_config['driver_order'])
    df['driver_code'] = df.driver_number.map(driver_config['driver_code'])
    
    df_layout = pd.read_csv(f'track_layout/{location}-{year}.csv')
    traces = []
    visibility = {driver_no: True if driver_code in team_groups[group_name] else "legendonly" for driver_code, driver_no in driver_config['driver_number'].items()}
    traces.append(go.Scatter(x=df_layout.x, y=df_layout.y, mode='lines', line=dict(dash='dot',color='#404040', width = 3), hoverinfo='skip', showlegend=False))
    
    for k, v in df.sort_values(by = ['driver_order']).groupby('driver_order'):
      traces.append(go.Scatter(x=v['x'], y=v['y'],
                               text=[f'{driver_config["driver_code"][v.driver_number.iloc[0]]}'],
                               textposition='top right',
                               textfont=dict(color=f'{driver_config["team_colour"][driver_config["driver_code"][v.driver_number.iloc[0]]]}'),
                               mode='markers+text',
                               marker={'size': 18,'color': f'{driver_config["team_colour"][driver_config["driver_code"][v.driver_number.iloc[0]]]}'},
                               name=f'{driver_config["driver_code"][v.driver_number.iloc[0]]}',
                               legendgroup=f'{driver_config["driver_code"][v.driver_number.iloc[0]]}',
                               visible=visibility[v.driver_number.iloc[0]]))

    layout = go.Layout(title = f'''Track Location {df.date.iloc[0]}''', xaxis=dict(title='X'), yaxis=dict(title='Y'), uirevision = 8, height=800, width=800, yaxis_range=[df_layout.y.min()-500,df_layout.y.max()+500], xaxis_range=[df_layout.x.min()-500,df_layout.x.max()+500])
    figure = go.Figure(data=traces, layout=layout)
    return figure

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
