import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def engineer_features(df):
    # Group by Year, Driver, Session and get best lap time
    best_laps = df.groupby(['Year', 'Driver', 'Session'])['LapTime'].min().reset_index()
    
    # Pivot to get FP1, FP2, FP3 times for each driver
    pivot_df = best_laps.pivot_table(index=['Year', 'Driver'], columns='Session', values='LapTime').reset_index()
    
    # Calculate improvements between sessions
    pivot_df['FP1_to_FP2_Improvement'] = pivot_df['FP1'] - pivot_df['FP2']
    pivot_df['FP2_to_FP3_Improvement'] = pivot_df['FP2'] - pivot_df['FP3']
    
    # Calculate consistency metrics
    pivot_df['FP_Consistency'] = pivot_df[['FP1', 'FP2', 'FP3']].std(axis=1)
    pivot_df['FP_Mean'] = pivot_df[['FP1', 'FP2', 'FP3']].mean(axis=1)
    
    # Calculate relative performance to fastest lap
    pivot_df['Relative_FP1'] = pivot_df['FP1'] / pivot_df.groupby('Year')['FP1'].transform('min')
    pivot_df['Relative_FP2'] = pivot_df['FP2'] / pivot_df.groupby('Year')['FP2'].transform('min')
    pivot_df['Relative_FP3'] = pivot_df['FP3'] / pivot_df.groupby('Year')['FP3'].transform('min')
    
    # Merge with original data to get team and other features
    pivot_df = pivot_df.merge(
        df[['Year', 'Driver', 'Team', 'Position', 'Stint', 'Compound', 'TrackStatus']].drop_duplicates(),
        on=['Year', 'Driver'],
        how='left'
    )
    
    # Fill missing values only for numeric columns
    numeric_cols = pivot_df.select_dtypes(include=[np.number]).columns
    pivot_df[numeric_cols] = pivot_df[numeric_cols].fillna(pivot_df[numeric_cols].median())
    
    return pivot_df

def train_xgboost_model(X, y):
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X, y)
    return model

def predict_2025_spanish_gp():
    # Load data
    df = pd.read_csv('spanish_gp_realdata.csv')
    
    # Convert LapTime to seconds if needed
    if pd.api.types.is_timedelta64_dtype(df['LapTime']):
        df['LapTime'] = df['LapTime'].dt.total_seconds()
    else:
        df['LapTime'] = pd.to_timedelta(df['LapTime']).dt.total_seconds()
    
    # Prepare features
    full_X = engineer_features(df)
    y = full_X['Position']
    
    # One-hot encode categorical columns
    categorical_cols = ['Compound', 'TrackStatus', 'Stint']
    X = pd.get_dummies(full_X, columns=categorical_cols, drop_first=True)
    
    # For model training, drop non-numeric columns
    X_model = X.drop(['Year', 'Driver', 'Team', 'Position'], axis=1)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_model)
    
    # Train model
    model = train_xgboost_model(X_scaled, y)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Driver': X['Driver'],
        'Team': X['Team'],
        'Predicted_Position': predictions,
        'Actual_2024_Position': y
    })
    
    # Sort by predicted position
    results = results.sort_values('Predicted_Position')
    
    # Save predictions
    results.to_csv('spanish_gp_2025_xgboost_predictions.csv', index=False)
    
    # Print top 5 predictions
    print("\n🏁 Predicted 2025 Spanish GP Top 5 🏁")
    print("=====================================")
    for _, row in results.head().iterrows():
        print(f"Driver: {row['Driver']}, Team: {row['Team']}")
        print(f"Predicted Position: {row['Predicted_Position']:.1f}")
        print(f"2024 Position: {row['Actual_2024_Position']}")
        print("-------------------------------------")
    
    # Calculate model error
    mae = np.mean(np.abs(predictions - y))
    print(f"\n🔍 Model Error (MAE): {mae:.2f} positions")
    
    # Plot feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_model.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance['Feature'], feature_importance['Importance'])
    plt.xticks(rotation=45, ha='right')
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance.png')
    plt.close()

if __name__ == "__main__":
    predict_2025_spanish_gp() 