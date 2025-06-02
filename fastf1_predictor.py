import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import fastf1
import fastf1.plotting
from datetime import datetime

# Enable FastF1 cache
fastf1.Cache.enable_cache('cache')

def get_driver_name(driver_number):
    # Map of driver numbers to names
    driver_map = {
        '1': 'Max Verstappen',
        '4': 'Lando Norris',
        '44': 'Lewis Hamilton',
        '63': 'George Russell',
        '16': 'Charles Leclerc',
        '55': 'Carlos Sainz',
        '81': 'Oscar Piastri',
        '11': 'Sergio Perez',
        '10': 'Pierre Gasly',
        '31': 'Esteban Ocon',
        '27': 'Nico Hulkenberg',
        '14': 'Fernando Alonso',
        '24': 'Zhou Guanyu',
        '18': 'Lance Stroll',
        '3': 'Daniel Ricciardo',
        '77': 'Valtteri Bottas',
        '20': 'Kevin Magnussen',
        '23': 'Alex Albon',
        '22': 'Yuki Tsunoda',
        '2': 'Logan Sargeant'
    }
    return driver_map.get(driver_number, f"Driver {driver_number}")

def load_fastf1_data():
    try:
        # Load 2024 Spanish GP data
        session = fastf1.get_session(2024, 'Spanish Grand Prix', 'R')
        session.load()
        
        # Get lap times and race data
        laps = session.laps
        drivers = session.drivers
        
        # Create DataFrame with relevant data
        data = []
        for driver in drivers:
            driver_laps = laps.pick_driver(driver)
            if len(driver_laps) > 0:
                # Get team name from driver info
                driver_info = session.get_driver(driver)
                team = driver_info.TeamName if hasattr(driver_info, 'TeamName') else 'Unknown'
                
                # Calculate race pace metrics
                best_lap = driver_laps['LapTime'].min().total_seconds()
                avg_lap = driver_laps['LapTime'].mean().total_seconds()
                last_lap = driver_laps['LapTime'].iloc[-1].total_seconds()
                
                # Calculate consistency
                lap_std = driver_laps['LapTime'].std().total_seconds()
                
                data.append({
                    'Driver': get_driver_name(driver),
                    'Team': team,
                    'BestLapTime': best_lap,
                    'AvgLapTime': avg_lap,
                    'LastLapTime': last_lap,
                    'LapTimeStd': lap_std,
                    'Position': driver_laps.iloc[-1]['Position'],
                    'TyreLife': driver_laps['TyreLife'].mean() if 'TyreLife' in driver_laps.columns else 0,
                    'Sector1Time': driver_laps['Sector1Time'].mean().total_seconds() if 'Sector1Time' in driver_laps.columns else 0,
                    'Sector2Time': driver_laps['Sector2Time'].mean().total_seconds() if 'Sector2Time' in driver_laps.columns else 0,
                    'Sector3Time': driver_laps['Sector3Time'].mean().total_seconds() if 'Sector3Time' in driver_laps.columns else 0,
                    'LapCount': len(driver_laps)
                })
        
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error loading FastF1 data: {str(e)}")
        return None

def engineer_features(df):
    if df is None:
        return None
        
    # Calculate relative performance metrics
    df['Relative_BestLap'] = df['BestLapTime'] / df['BestLapTime'].min()
    df['Relative_AvgLap'] = df['AvgLapTime'] / df['AvgLapTime'].min()
    df['Relative_LastLap'] = df['LastLapTime'] / df['LastLapTime'].min()
    
    # Calculate sector performance if available
    if 'Sector1Time' in df.columns:
        df['Sector1_Relative'] = df['Sector1Time'] / df['Sector1Time'].min()
        df['Sector2_Relative'] = df['Sector2Time'] / df['Sector2Time'].min()
        df['Sector3_Relative'] = df['Sector3Time'] / df['Sector3Time'].min()
    else:
        df['Sector1_Relative'] = df['Relative_BestLap']
        df['Sector2_Relative'] = df['Relative_BestLap']
        df['Sector3_Relative'] = df['Relative_BestLap']
    
    # Calculate consistency metrics
    df['LapTime_Consistency'] = df['LapTimeStd'] / df['AvgLapTime']
    df['Race_Pace'] = df['AvgLapTime'] / df['BestLapTime']
    df['Endurance'] = df['LastLapTime'] / df['BestLapTime']
    
    return df

def train_model(X, y):
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    model.fit(X, y)
    return model

def predict_2025_spanish_gp():
    # Load and prepare data
    df = load_fastf1_data()
    if df is None:
        print("Failed to load FastF1 data. Please check your internet connection and try again.")
        return
        
    df = engineer_features(df)
    
    # Prepare features for training
    feature_cols = [
        'Relative_BestLap', 'Relative_AvgLap', 'Relative_LastLap',
        'Sector1_Relative', 'Sector2_Relative', 'Sector3_Relative',
        'LapTime_Consistency', 'Race_Pace', 'Endurance', 'TyreLife'
    ]
    
    X = df[feature_cols]
    y = df['Position']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = train_model(X_scaled, y)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Driver': df['Driver'],
        'Team': df['Team'],
        'Predicted_Position': predictions,
        'Actual_2024_Position': df['Position'],
        'Best_Lap': df['BestLapTime'].round(3),
        'Avg_Lap': df['AvgLapTime'].round(3),
        'Consistency': (1 - df['LapTime_Consistency']).round(3)  # Higher is better
    })
    
    # Sort by predicted position
    results = results.sort_values('Predicted_Position')
    
    # Save predictions
    results.to_csv('spanish_gp_2025_fastf1_predictions.csv', index=False)
    
    # Print predictions
    print("\n🏁 Predicted 2025 Spanish GP Results 🏁")
    print("=====================================")
    for _, row in results.iterrows():
        print(f"Driver: {row['Driver']}")
        print(f"Team: {row['Team']}")
        print(f"Predicted Position: {row['Predicted_Position']:.1f}")
        print(f"2024 Position: {row['Actual_2024_Position']}")
        print(f"Best Lap: {row['Best_Lap']}s")
        print(f"Avg Lap: {row['Avg_Lap']}s")
        print(f"Consistency: {row['Consistency']:.3f}")
        print("-------------------------------------")
    
    # Calculate model error
    mae = np.mean(np.abs(predictions - y))
    print(f"\n🔍 Model Error (MAE): {mae:.2f} positions")
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n📊 Feature Importance:")
    print("=====================")
    for _, row in feature_importance.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.3f}")

if __name__ == "__main__":
    predict_2025_spanish_gp() 