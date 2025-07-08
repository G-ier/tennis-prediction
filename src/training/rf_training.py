from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import sys
import os

# Add the root directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# Use absolute import instead of relative import
from src.data_processing.tns_processing import TNSData
import joblib
from tqdm import tqdm

def train_random_forest(X_train, y_train, X_test=None, y_test=None, n_estimators=120, max_depth=None, save_path=None):
    """
    Train a Random Forest classifier
    
    Parameters:
    -----------
    X_train : training features
    y_train : training labels
    X_test : test features (optional)
    y_test : test labels (optional)
    n_estimators : number of trees in the forest
    max_depth : maximum depth of each tree
    save_path : path to save the model (optional)
    
    Returns:
    --------
    rf_model : trained RandomForestClassifier
    accuracy : test accuracy (if test data provided)
    """
    # Initialize the model
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        verbose=1  # Enable verbose output for progress tracking
    )
    
    # Train the model with a progress message
    print("Training Random Forest model...")
    rf_model.fit(X_train, y_train)
    print("Training completed!")
    
    # Evaluate if test data is provided
    accuracy = None
    if X_test is not None and y_test is not None:
        print("Evaluating model on test data...")
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        print("\nTop 10 important features:")
        print(feature_importance.head(10))
    
    # Save model if path is provided
    if save_path is not None:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(rf_model, save_path)
        print(f"Model saved to {save_path}")
    
    return rf_model, accuracy

def main():
    # Use an absolute path based on the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../..'))
    data_path = os.path.join(project_root, 'data', 'atp_short.csv')
    
    tns_data = TNSData(data_path)

    # Split data first
    train_data, test_data = tns_data.data_split()

    # Load training data
    train_data, train_labels = tns_data.process_for_training_RF(train_data)

    # Load test data
    test_data, test_labels = tns_data.process_for_testing_RF(test_data)

    # Train model and save it
    rf_model, accuracy = train_random_forest(
        train_data, train_labels, 
        test_data, test_labels,
        save_path="models/random_forest_model.joblib"
    )

if __name__ == "__main__":
    main()  