# To do 

## Dash app 
  - [ ] LAP NUMBER IN TELEMETRY & LAPTIMES ARE OFFSET BY 1(nth lap in laptimes is ahead in time compared to telemtry)
  - [ ] organise plot layouts
  - [ ] public hosting???
  - [ ] add live time delta plots 
  - [x] add live position table
  - [ ] Submit button
  
### Laptimes plot 
  - [ ] toggle laptimes by teams, useful for car level comparison
  - [x] add filter button 
### Telemetry plot 
  - [ ] 2 separate calls to sql table in telemetry plot, do 1 query & split in function (repeat for other suitable tables)
  - [ ] Laptime on top of plot isn't matched with the plot, might be printing a different lap time
  - [x] replace drop down with buttons 
  - [x] Differentiate drivers from same team 
### Tables 
  - [ ] Conditional formatting for different tables
  - [ ] improve table ui
  

## Core logic 
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
  - [ ] SAving runtime states - update dB
  - [ ] Driver grouping 
  - [ ] Pit duration and stationary time

# Updates:

 - added track positions
 - added driver positions
 - added laptime filter
 - fixed button logic, might have to scroll right for later lap buttons

 
