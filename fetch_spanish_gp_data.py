import fastf1
import pandas as pd
import os

# Years to fetch
YEARS = [2021, 2022, 2023, 2024]
EVENT_NAME = 'Spanish Grand Prix'

all_data = []

for year in YEARS:
    print(f"Fetching data for {year}...")
    schedule = fastf1.get_event_schedule(year)
    spain = schedule[schedule['EventName'] == EVENT_NAME]
    if spain.empty:
        print(f"No Spanish GP found for {year}")
        continue
    gp_round = int(spain.iloc[0]['RoundNumber'])
    
    # Fetch sessions
    for session_name in ['FP1', 'FP2', 'FP3', 'Q', 'R']:
        try:
            session = fastf1.get_session(year, gp_round, session_name)
            session.load()
            laps = session.laps
            if laps.empty:
                continue
            # Get driver, team, best lap time, session type
            best_laps = laps.groupby('Driver').apply(lambda x: x.loc[x['LapTime'].idxmin()])
            best_laps = best_laps.reset_index(drop=True)
            best_laps['Session'] = session_name
            best_laps['Year'] = year
            all_data.append(best_laps)
        except Exception as e:
            print(f"Failed to fetch {session_name} for {year}: {e}")

if all_data:
    df = pd.concat(all_data, ignore_index=True)
    # Select relevant columns
    df = df[['Year', 'Session', 'Driver', 'Team', 'LapTime', 'Position', 'Stint', 'Compound', 'TrackStatus', 'IsPersonalBest']]
    # Save to CSV
    df.to_csv('spanish_gp_realdata.csv', index=False)
    print("Saved real Spanish GP data to spanish_gp_realdata.csv")
else:
    print("No data fetched.") 