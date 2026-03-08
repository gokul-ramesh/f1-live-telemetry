# Project status / To do

Detailed checklist and changelog. See [README.md](README.md) for project overview and usage.

---

## To do

### Dash app

- [x] highlight top 10 in position table
- [ ] add position vs lap plot
- [x] add track position graph
- [ ] filter out blue flags in race control table?
- [x] race control table
- [ ] add team radio table
- [x] LAP NUMBER IN TELEMETRY & LAPTIMES ARE OFFSET BY 1 (nth lap in laptimes is ahead in time compared to telemetry) - update_db_historical.py Line 88, Assign [0,0]
- [ ] organise plot layouts
- [ ] public hosting???
- [ ] add live time delta plots
- [x] add live position table
- [x] Submit button

#### Laptimes plot

- [ ] fix live laptimes fetch
- [x] toggle laptimes by teams, useful for car level comparison
- [x] add filter button

#### Telemetry plot

- [ ] 2 separate calls to sql table in telemetry plot, do 1 query & split in function (repeat for other suitable tables)
- [ ] Laptime on top of plot isn't matched with the plot, might be printing a different lap time
- [x] replace drop down with buttons
- [x] Differentiate drivers from same team

#### Tables

- [x] Conditional formatting for different tables
- [ ] improve table ui

### Core logic

- [ ] memoise dash computations [link](https://dash.plotly.com/performance)
- [x] add gap to leader table
- [ ] add lap number adjust logic
- [x] Update lap number counter to add last recorded distance
- [x] ~~Optimize start line detection~~ Moved to config file
- [ ] add live time delta
- [ ] fuel corrected laptime plots
- [ ] more controllable/modifiable lap assignments logic? if we spot an error, can we alter the offset for each driver in the dash ui
- [ ] tyre compound data?
- [ ] separate dashes/dash tabs for quali/race session, might need different plots
- [x] fetch live position from API
- [x] Add config files for corners, circuit length and start lines
- [ ] Saving runtime states - update_db
- [x] Driver grouping
- [ ] Pit duration and stationary time
- [ ] Historical analysis to calculate tyre deg

---

## Updates / Changelog

- added track positions
- added driver positions
- added laptime filter
- fixed button logic, might have to scroll right for later lap buttons
- added fastest lap times of drivers and few latest lap times
