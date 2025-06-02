import fastf1
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories for data storage."""
    directories = ['data', 'cache', 'models', 'notebooks']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    logger.info("Created necessary directories")

# Create directories first
setup_directories()

# Enable FastF1 cache
fastf1.Cache.enable_cache('cache')

def fetch_session_data(year, gp, session_type):
    """Fetch session data for a specific Grand Prix."""
    try:
        session = fastf1.get_session(year, gp, session_type)
        session.load()
        return session
    except Exception as e:
        logger.error(f"Error fetching {session_type} data for {year} {gp}: {str(e)}")
        return None

def extract_lap_features(session):
    """Extract relevant features from lap data."""
    if not session:
        return None
    
    laps = session.laps
    drivers = session.drivers
    
    features = []
    for driver in drivers:
        driver_laps = laps.pick_driver(driver)
        if len(driver_laps) == 0:
            continue
            
        # Basic lap statistics
        avg_lap_time = driver_laps['LapTime'].mean().total_seconds()
        best_lap_time = driver_laps['LapTime'].min().total_seconds()
        lap_count = len(driver_laps)
        
        # Stint analysis
        stints = driver_laps[['Stint', 'Compound', 'LapTime']].groupby('Stint').agg({
            'Compound': 'first',
            'LapTime': ['count', 'mean']
        })
        
        # Weather data
        weather_data = session.weather_data
        avg_track_temp = weather_data['TrackTemp'].mean()
        avg_air_temp = weather_data['AirTemp'].mean()
        
        features.append({
            'Driver': driver,
            'AvgLapTime': avg_lap_time,
            'BestLapTime': best_lap_time,
            'LapCount': lap_count,
            'AvgTrackTemp': avg_track_temp,
            'AvgAirTemp': avg_air_temp,
            'StintCount': len(stints)
        })
    
    return pd.DataFrame(features)

def process_historical_data(start_year=2021, end_year=2024):
    """Process historical data for model training."""
    all_data = []
    
    for year in range(start_year, end_year + 1):
        logger.info(f"Processing data for year {year}")
        
        # Get list of Grands Prix for the year
        schedule = fastf1.get_event_schedule(year)
        
        for _, event in schedule.iterrows():
            gp = event['EventName']
            logger.info(f"Processing {gp}")
            
            # Fetch practice sessions
            fp1_data = fetch_session_data(year, gp, 'FP1')
            fp2_data = fetch_session_data(year, gp, 'FP2')
            fp3_data = fetch_session_data(year, gp, 'FP3')
            
            # Extract features for each session
            fp1_features = extract_lap_features(fp1_data)
            fp2_features = extract_lap_features(fp2_data)
            fp3_features = extract_lap_features(fp3_data)
            
            if fp1_features is not None and fp2_features is not None and fp3_features is not None:
                # Combine features
                combined_features = pd.merge(
                    fp1_features, fp2_features, 
                    on='Driver', 
                    suffixes=('_FP1', '_FP2')
                )
                combined_features = pd.merge(
                    combined_features, fp3_features,
                    on='Driver',
                    suffixes=('', '_FP3')
                )
                
                # Add metadata
                combined_features['Year'] = year
                combined_features['GrandPrix'] = gp
                
                all_data.append(combined_features)
    
    # Combine all data
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv('data/historical_data.csv', index=False)
        logger.info("Historical data processing completed")
        return final_df
    else:
        logger.error("No data was processed")
        return None

def main():
    """Main function to run the data fetching process."""
    logger.info("Starting data fetching process")
    
    # Process historical data
    historical_data = process_historical_data()
    
    if historical_data is not None:
        logger.info(f"Successfully processed data for {len(historical_data)} sessions")
    else:
        logger.error("Failed to process historical data")

if __name__ == "__main__":
    main() 