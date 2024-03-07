import matplotlib.pyplot as plt
import matplotlib.dates as dates
import json
from datetime import datetime, timedelta

from urllib.request import urlopen

response = urlopen('https://api.openf1.org/v1/car_data?driver_number=1&session_key=latest')
throttles_ver = json.loads(response.read().decode('utf-8'))

response = urlopen('https://api.openf1.org/v1/car_data?driver_number=16&session_key=latest')
throttles_lec = json.loads(response.read().decode('utf-8'))

'''
i=1
laps=[]
while i<25:
    response = urlopen(f'https://api.openf1.org/v1/laps?session_key=latest&driver_number=16&lap_number={i}')
    try:
        laps.append(json.loads(response.read().decode('utf-8'))[0])
    except:
        break
    i+=1
'''
response = urlopen(f'https://api.openf1.org/v1/laps?session_key=latest&driver_number=1&lap_number=5')
laps_ver = json.loads(response.read().decode('utf-8'))

response = urlopen(f'https://api.openf1.org/v1/laps?session_key=latest&driver_number=16&lap_number=9')
laps_lec = json.loads(response.read().decode('utf-8'))

#throttles = json.load(open("/home/gokul/Documents/f1/ver_car.json"))
#laps = json.load(open("/home/gokul/Documents/f1/ver_lap.json"))

xax = []
yax_ver = []
yax_lec = []

start_time_lec = datetime.strptime(laps_lec[0]["date_start"], "%Y-%m-%dT%H:%M:%S.%f")
end_time_lec = start_time_lec + timedelta(milliseconds=laps_lec[0]['lap_duration']*1000)

start_time_ver = datetime.strptime(laps_ver[0]["date_start"], "%Y-%m-%dT%H:%M:%S.%f")
end_time_ver = start_time_ver + timedelta(milliseconds=laps_ver[0]['lap_duration']*1000)


isLap = False
for thr in throttles_ver:
    t = datetime.strptime(thr['date'], "%Y-%m-%dT%H:%M:%S.%f")
    if t >= start_time_ver or isLap:
        isLap = True
        if t <= end_time_ver:
            #xax.append(t)
            yax_ver.append(thr["throttle"])
        else:
            break

isLap = False
for thr in throttles_lec:
    t = datetime.strptime(thr['date'], "%Y-%m-%dT%H:%M:%S.%f")
    if t >= start_time_lec or isLap:
        isLap = True
        if t <= end_time_lec:
            #xax.append(t)
            yax_lec.append(thr["throttle"])
        else:
            break

print(len(yax_ver), len(yax_lec))

plt.plot(list(range(len(yax_lec))), yax_lec, label="LEC")
plt.plot(list(range(len(yax_ver))), yax_ver, label="VER")
plt.legend()

plt.show()

