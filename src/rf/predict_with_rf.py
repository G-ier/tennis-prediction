import os
import joblib
import pandas as pd
import numpy as np
import sys

# Add the project root directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.append(project_root)

from src.data_processing.tns_processing import TNSData

def main():
    # Define paths
    model_path = os.path.join(project_root, 'src/training/models/random_forest_model.joblib')
    data_path = os.path.join(project_root, 'data/atp_real.csv')
    
    # Load the model
    print("Loading the RandomForest model...")
    rf_model = joblib.load(model_path)
    
    # Load and process the data
    print("Loading and processing data...")
    tns_data = TNSData(data_path)
    
    # Process data for prediction (using the same function as for testing)
    X_predict = tns_data.process_for_evaluation()

    # Process data for second base predictions
    X_base_predictions = tns_data.process_for_evaluation_backup()
    
    # Make predictions
    print("Making predictions...")
    predictions = rf_model.predict(X_predict)
    probabilities = rf_model.predict_proba(X_predict)

    #  Make second base predictions
    base_predictions = rf_model.predict(X_base_predictions)
    base_probabilities = rf_model.predict_proba(X_base_predictions)
    
    # Create a results DataFrame
    results = pd.DataFrame()
    
    # Read the original data to get player information
    print("Loading original data for additional info...")
    original_data = pd.read_csv(data_path)
    
    # Include original players info if available
    if 'Player_1' in original_data.columns and 'Player_2' in original_data.columns:
        results['Player_1'] = original_data['Player_1']
        results['Player_2'] = original_data['Player_2']
    
    # Add predictions - 1 means Player_1 wins, 0 means Player_2 wins
    results['Predicted_Winner'] = predictions
    results['Prediction_Confidence'] = np.max(probabilities, axis=1)

    # Add base predictions
    results['Base_Predictions'] = base_predictions
    results['Base_Probabilities'] = np.max(base_probabilities, axis=1)
    
    # Map predictions back to player names
    results['Predicted_Winner_Name'] = results.apply(
        lambda row: row['Player_1'] if row['Predicted_Winner'] == 1 else row['Player_2'],
        axis=1
    )
    results['Base_Predicted_Winner_Name'] = results.apply(
        lambda row: row['Player_1'] if row['Base_Predictions'] == 1 else row['Player_2'],
        axis=1
    )
    
    # Save results
    output_path = os.path.join(project_root, 'data/rf_predictions.csv')
    results.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main() 