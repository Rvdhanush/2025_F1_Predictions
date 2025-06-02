import pandas as pd
import numpy as np

# Load preprocessed FP1–FP3 data (assume same format as used in f1_predictor.py)
df = pd.read_csv('spanish_gp_realdata.csv')

# Convert LapTime to seconds if needed
if pd.api.types.is_timedelta64_dtype(df['LapTime']):
    df['LapTime'] = df['LapTime'].dt.total_seconds()
else:
    df['LapTime'] = pd.to_timedelta(df['LapTime']).dt.total_seconds()

# Get only FP1–FP3 sessions for 2025 prediction (use 2024 as proxy for now)
practice = df[(df['Year'] == 2024) & (df['Session'].isin(['FP1', 'FP2', 'FP3']))]

# Team and driver multipliers (example values, can be tuned)
team_multipliers = {
    'McLaren': 1.10,
    'Ferrari': 1.08,
    'Mercedes': 1.06,
    'Red Bull': 1.05,
    'Aston Martin': 1.03,
    'Alpine': 1.01,
    'Williams': 1.00,
    'Kick Sauber': 1.00,
    'RB': 0.98,
    'Haas': 0.97
}
driver_multipliers = {
    'PIA': 1.10,  # Piastri
    'NOR': 1.09,  # Norris
    'LEC': 1.08,  # Leclerc
    'RUS': 1.07,  # Russell
    'HUL': 1.06,  # Hulkenberg
    'VER': 1.08,  # Verstappen
    'HAM': 1.07,  # Hamilton
    'SAI': 1.06,  # Sainz
    'PER': 1.05,  # Perez
    'ALO': 1.05   # Alonso
}

# Track specialist bonus (example: Verstappen, Norris)
specialist_bonus = {
    'VER': 0.02,
    'NOR': 0.01,
    'PIA': 0.01
}

# Home driver bonus (example: Sainz, Alonso)
home_bonus = {
    'SAI': 0.02,
    'ALO': 0.02
}

def score_driver(driver, team, laps):
    # Best lap (lower is better)
    best_lap = laps['LapTime'].min()
    # Long-run pace (mean of all laps)
    long_run = laps['LapTime'].mean()
    # Progression (FP1 to FP3 improvement)
    fp1 = laps[laps['Session'] == 'FP1']['LapTime'].min() if 'FP1' in laps['Session'].values else np.nan
    fp3 = laps[laps['Session'] == 'FP3']['LapTime'].min() if 'FP3' in laps['Session'].values else np.nan
    progression = fp1 - fp3 if pd.notnull(fp1) and pd.notnull(fp3) else 0
    # Consistency (std dev of laps)
    consistency = laps['LapTime'].std()
    # Base score: weighted sum (lower lap times and higher progression are better)
    score = (
        (1 / best_lap) * 0.4 +
        (1 / long_run) * 0.2 +
        (progression / fp1) * 0.2 if fp1 else 0 +
        (1 / (1 + consistency)) * 0.2
    )
    # Apply multipliers
    score *= team_multipliers.get(team, 1.0)
    score *= driver_multipliers.get(driver, 1.0)
    score += specialist_bonus.get(driver, 0)
    score += home_bonus.get(driver, 0)
    return score

# Score all drivers
results = []
for driver in practice['Driver'].unique():
    team = practice[practice['Driver'] == driver]['Team'].iloc[0]
    laps = practice[practice['Driver'] == driver]
    score = score_driver(driver, team, laps)
    results.append({'Driver': driver, 'Team': team, 'Score': score})

# Sort by score (higher is better)
pred_df = pd.DataFrame(results).sort_values('Score', ascending=False)
pred_df.reset_index(drop=True, inplace=True)

# Output top 5
print('Logic-Based Model: Top 5 Predicted Finishers (2025 Spanish GP)')
for i, row in pred_df.head(5).iterrows():
    print(f"{i+1}. {row['Driver']} ({row['Team']}) - Score: {row['Score']:.4f}")

# Save predictions
pred_df.to_csv('spanish_gp_2025_logic_predictions.csv', index=False) 