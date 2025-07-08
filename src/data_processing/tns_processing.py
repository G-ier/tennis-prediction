import os
import numpy as np
import pandas as pd
from src.data_processing.tennis_elo import load_data, compute_elo, get_recent_performance


class TNSData():
    def __init__(self, dir):
        self.data_dir = dir
        pass
        
    def recieve_kaggle_data(self):
        pass
    
    def _load_data(self):
        pass

    def load_elo_data(self):
        # Get absolute path to the project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '../..'))
        elo_data_path = os.path.join(project_root, 'data', 'atp_tennis.csv')

        # load elo calculation data with absolute path
        matches = load_data(elo_data_path, row_limit=56361)

    def data_split(self):
        """
        Split the data into training (90%) and testing (10%) sets
        
        Returns:
        --------
        train_data : DataFrame containing 80% of the data for training
        test_data : DataFrame containing 20% of the data for testing
        """

        # Get absolute path to the project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '../..'))
        elo_data_path = os.path.join(project_root, 'data', 'atp_tennis.csv')

        performance_data_path = os.path.join(project_root, 'data', 'atp_short.csv')

        # load elo calculation data with absolute path
        matches = load_data(elo_data_path, row_limit=64781)
        print(f"Loaded {len(matches)} matches for ELO calculation")
        
        # Compute Elo ratings (overall and court-specific)
        players_to_elo, court_elo = compute_elo(matches)

        # load performance data with absolute path
        performance_data = load_data(performance_data_path, row_limit=8421)
        print(f"Loaded {len(performance_data)} matches for performance calculation")

        matches_subset, player_stats_df = get_recent_performance(performance_data, row_limit=6735, recent_matches=25)

        # Load the raw data
        raw_data = pd.read_csv(self.data_dir)
        
        # Add player statistics to raw data - only keep win_percentage
        for player_col in ['Player_1', 'Player_2']:
            # Create a simplified player stats DataFrame with just player and win_percentage
            simple_stats = player_stats_df[['player', 'win_percentage']]
            
            # Merge player statistics for each player
            raw_data = raw_data.merge(
                simple_stats,
                left_on=player_col,
                right_on='player',
                how='left',
                suffixes=('', f'_{player_col}')
            )
            # Rename the win_percentage column to indicate which player it belongs to
            raw_data = raw_data.rename(columns={'win_percentage': f'win_percentage_{player_col}'})
            # Fill NaN values with 0 for players not found in statistics
            raw_data[f'win_percentage_{player_col}'] = raw_data[f'win_percentage_{player_col}'].fillna(0)
            # Drop the player column from the merge
            raw_data = raw_data.drop(columns=['player'])

        print("Added player win percentages to raw data (0 for unknown players)")
        
        # Add ELO ratings for each player
        for player_col in ['Player_1', 'Player_2']:

            # Add overall ELO ratings
            #raw_data[f'{player_col}_elo'] = raw_data[player_col].apply(
             #   lambda player: players_to_elo.get(player, pd.DataFrame({'ranking': [1500]}))['ranking'].iloc[-1] 
             #  if player in players_to_elo else 1500
            #)
            
            # Add court-specific ELO ratings
            #for surface in ['Hard', 'Clay', 'Grass', 'Carpet']:

            
            # Add court-specific ELO ratings based on the match surface
            raw_data[f'{player_col}_elo_surface'] = raw_data.apply(
                lambda row: court_elo.get(row['Surface'], {}).get(row[player_col], pd.DataFrame({'ranking': [1500]}))['ranking'].iloc[-1]
                if row['Surface'] in court_elo and row[player_col] in court_elo[row['Surface']] else 1500,
                axis=1
            )
        
        print("Added Court-specific ELO ratings and recent performance to raw data")
        
        # Create one-hot encoding for categorical variables before splitting
        # This ensures both train and test data have the same columns -> somehow we get missing columns in the test data
        if 'Surface' in raw_data.columns:
            raw_data = pd.get_dummies(raw_data, columns=['Surface'])
        if 'Court' in raw_data.columns:
            raw_data = pd.get_dummies(raw_data, columns=['Court'])
        
        # Calculate the split point
        num_rows = len(raw_data)
        num_rows_80_percent = int(num_rows * 0.8)
        
        # Split the data
        train_data = raw_data.iloc[:num_rows_80_percent]
        test_data = raw_data.iloc[num_rows_80_percent:]
        
        print(f"Data split: {len(train_data)} training samples, {len(test_data)} testing samples")
        
        return train_data, test_data

    def process_for_training_RF(self, raw_data):
        # Create a single mapping for all players - using element-wise comparison
        raw_data['winner'] = (raw_data['Winner'] == raw_data['Player_1']).astype(int)

        # Define columns to drop
        columns_to_drop = [
            'Date', 'Series', 'Tournament', 'Round', 'Pts_1', 'Pts_2', 
            'Score', 'Best of', 'Player_1', 'Player_2', 'Winner',
            'player',  # Drop the player column from player_stats_df
            'player_Player_1', 'player_Player_2'  # Drop the player columns from the merged stats
        ]
        
        # Only drop columns that exist in the DataFrame
        existing_columns = [col for col in columns_to_drop if col in raw_data.columns]
        raw_data = raw_data.drop(columns=existing_columns)

        # Return a tuple of the data and the winner column
        return (raw_data.drop(columns=['winner']), raw_data['winner'])
    
    def process_for_testing_RF(self, raw_data):
        # Create a single mapping for all players - using element-wise comparison
        raw_data['winner'] = (raw_data['Winner'] == raw_data['Player_1']).astype(int)

        # Define columns to drop
        columns_to_drop = [
            'Date', 'Series', 'Tournament', 'Round', 'Pts_1', 'Pts_2', 
            'Score', 'Best of', 'Player_1', 'Player_2', 'Winner',
            'player',  # Drop the player column from player_stats_df
            'player_Player_1', 'player_Player_2'  # Drop the player columns from the merged stats
        ]
        
        # Only drop columns that exist in the DataFrame
        existing_columns = [col for col in columns_to_drop if col in raw_data.columns]
        raw_data = raw_data.drop(columns=existing_columns)

        # Return a tuple of the data and the winner column
        return (raw_data.drop(columns=['winner']), raw_data['winner'])

    def process_for_evaluation(self):
        """
        Process data for evaluation/prediction with the RandomForest model.
        Uses actual ELO ratings and player statistics from historical data.
        
        Returns:
        --------
        DataFrame: Processed data with features in the expected order
        """
        # Get absolute path to the project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '../..'))
        elo_data_path = os.path.join(project_root, 'data', 'atp_tennis.csv')
        performance_data_path = os.path.join(project_root, 'data', 'atp_short.csv')

        # Load ELO calculation data
        matches = load_data(elo_data_path, row_limit=64781)
        print(f"Loaded {len(matches)} matches for ELO calculation")
        
        # Compute Elo ratings (overall and court-specific)
        players_to_elo, court_elo = compute_elo(matches)

        # Load performance data
        performance_data = load_data(performance_data_path, row_limit=8421)
        print(f"Loaded {len(performance_data)} matches for performance calculation")

        # Calculate recent performance metrics
        matches_subset, player_stats_df = get_recent_performance(performance_data, row_limit=6735, recent_matches=25)

        # Load the raw data for prediction
        print("Loading raw data...")
        raw_data = pd.read_csv(self.data_dir)
        
        # Add player statistics to raw data - only keep win_percentage
        for player_col in ['Player_1', 'Player_2']:
            # Create a simplified player stats DataFrame with just player and win_percentage
            simple_stats = player_stats_df[['player', 'win_percentage']]
            
            # Merge player statistics for each player
            raw_data = raw_data.merge(
                simple_stats,
                left_on=player_col,
                right_on='player',
                how='left',
                suffixes=('', f'_{player_col}')
            )
            # Rename the win_percentage column to indicate which player it belongs to
            raw_data = raw_data.rename(columns={'win_percentage': f'win_percentage_{player_col}'})
            # Fill NaN values with 0.5 for players not found in statistics
            raw_data[f'win_percentage_{player_col}'] = raw_data[f'win_percentage_{player_col}'].fillna(0.5)
            # Drop the player column from the merge
            raw_data = raw_data.drop(columns=['player'])

        print("Added player win percentages to raw data")
        
        # Add ELO ratings for each player
        for player_col in ['Player_1', 'Player_2']:
            # Add court-specific ELO ratings based on the match surface
            raw_data[f'{player_col}_elo_surface'] = raw_data.apply(
                lambda row: court_elo.get(row['Surface'], {}).get(row[player_col], pd.DataFrame({'ranking': [1500]}))['ranking'].iloc[-1]
                if row['Surface'] in court_elo and row[player_col] in court_elo[row['Surface']] else 1500,
                axis=1
            )
        
        print("Added Court-specific ELO ratings")
        
        # Create one-hot encoding for categorical variables
        if 'Surface' in raw_data.columns:
            raw_data = pd.get_dummies(raw_data, columns=['Surface'], prefix='Surface')
        if 'Court' in raw_data.columns:
            raw_data = pd.get_dummies(raw_data, columns=['Court'], prefix='Court')
            
        # Ensure all expected features are present
        required_features = ['Surface_Clay', 'Surface_Grass', 'Surface_Hard', 'Court_Indoor', 'Court_Outdoor']
        for feature in required_features:
            if feature not in raw_data.columns:
                prefix, value = feature.split('_')
                if prefix == 'Surface' and any(col.startswith('Surface_') for col in raw_data.columns):
                    # Only add if we have other Surface columns but missing this one
                    print(f"Adding missing feature: {feature}")
                    raw_data[feature] = 0
                elif prefix == 'Court' and any(col.startswith('Court_') for col in raw_data.columns):
                    # Only add if we have other Court columns but missing this one
                    print(f"Adding missing feature: {feature}")
                    raw_data[feature] = 0

        # Define columns to drop
        columns_to_drop = [
            'Date', 'Series', 'Tournament', 'Round', 'Pts_1', 'Pts_2', 
            'Score', 'Best of', 'Player_1', 'Player_2', 'Winner',
            'player',  # Drop the player column from player_stats_df
            'player_Player_1', 'player_Player_2'  # Drop the player columns from the merged stats
        ]
        
        # Only drop columns that exist in the DataFrame
        existing_columns = [col for col in columns_to_drop if col in raw_data.columns]
        raw_data = raw_data.drop(columns=existing_columns)

        # Expected feature columns in the correct order
        expected_features = [
            'Rank_1', 'Rank_2', 'Odd_1', 'Odd_2', 
            'win_percentage_Player_1', 'win_percentage_Player_2',
            'Player_1_elo_surface', 'Player_2_elo_surface',
            'Surface_Clay', 'Surface_Grass', 'Surface_Hard',
            'Court_Indoor', 'Court_Outdoor'
        ]
        
        # Ensure all expected columns exist with appropriate defaults
        for col in expected_features:
            if col not in raw_data.columns:
                if col in ['Rank_1', 'Rank_2']:
                    # Use a reasonable default ranking
                    print(f"Adding missing feature: {col}")
                    raw_data[col] = 50
                elif col in ['Odd_1', 'Odd_2']:
                    # Use even odds as default
                    print(f"Adding missing feature: {col}")
                    raw_data[col] = 2.0
                else:
                    # For other features, default to 0
                    print(f"Adding missing feature: {col}")
                    raw_data[col] = 0
        
        # Return only the expected features in the correct order
        print(f"Prepared data with {len(expected_features)} features in the correct order")
        return raw_data[expected_features]

    def process_for_evaluation_backup(self):
        """
        Backup version of process_for_evaluation that uses default values.
        Ensures all expected features are present and in the correct order.
        
        Returns:
        --------
        DataFrame: Processed data with features in the expected order
        """
        # Load the raw data
        print("Loading raw data...")
        raw_data = pd.read_csv(self.data_dir)
        
        # Default ELO surface ratings and win percentages for missing values
        print("Setting up default features...")
        raw_data['Player_1_elo_surface'] = 1500
        raw_data['Player_2_elo_surface'] = 1500
        raw_data['win_percentage_Player_1'] = 0.5
        raw_data['win_percentage_Player_2'] = 0.5
        
        # Handle Surface one-hot encoding
        if 'Surface' in raw_data.columns:
            print("Processing Surface feature...")
            # Create dummy variables for Surface
            surface_dummies = pd.get_dummies(raw_data['Surface'], prefix='Surface')
            
            # Ensure all expected Surface columns exist
            for surface in ['Clay', 'Grass', 'Hard']:
                col_name = f'Surface_{surface}'
                if col_name not in surface_dummies.columns:
                    surface_dummies[col_name] = 0
            
            # Add dummy columns to the DataFrame
            raw_data = pd.concat([raw_data, surface_dummies], axis=1)
        else:
            # If Surface column doesn't exist, add default values
            raw_data['Surface_Clay'] = 0
            raw_data['Surface_Grass'] = 0
            raw_data['Surface_Hard'] = 1  # Default to Hard surface
        
        # Handle Court one-hot encoding
        if 'Court' in raw_data.columns:
            print("Processing Court feature...")
            # Create dummy variables for Court
            court_dummies = pd.get_dummies(raw_data['Court'], prefix='Court')
            
            # Ensure all expected Court columns exist
            for court in ['Indoor', 'Outdoor']:
                col_name = f'Court_{court}'
                if col_name not in court_dummies.columns:
                    court_dummies[col_name] = 0
            
            # Add dummy columns to the DataFrame
            raw_data = pd.concat([raw_data, court_dummies], axis=1)
        else:
            # If Court column doesn't exist, add default values
            raw_data['Court_Indoor'] = 0
            raw_data['Court_Outdoor'] = 1  # Default to Outdoor court
        
        # Expected feature columns in the correct order
        expected_features = [
            'Rank_1', 'Rank_2', 'Odd_1', 'Odd_2', 
            'win_percentage_Player_1', 'win_percentage_Player_2',
            'Player_1_elo_surface', 'Player_2_elo_surface',
            'Surface_Clay', 'Surface_Grass', 'Surface_Hard',
            'Court_Indoor', 'Court_Outdoor'
        ]
        
        # Ensure all expected columns exist
        for col in expected_features:
            if col not in raw_data.columns:
                if col in ['Rank_1', 'Rank_2', 'Odd_1', 'Odd_2']:
                    # Fill with a reasonable default for these features
                    raw_data[col] = 50
                else:
                    # For other features, default to 0
                    raw_data[col] = 0
        
        # Return only the expected features in the correct order
        print(f"Prepared data with {len(expected_features)} features in the correct order")
        return raw_data[expected_features]

    def process_for_Transformer(self):
        
        # Get raw data with ELO ratings from data_split method
        raw_data, _ = self.data_split()

        # generate new columns for players
        # Add player columns
        raw_data['player_1'] = raw_data['Player_1'].fillna('Unknown')
        raw_data['player_2'] = raw_data['Player_2'].fillna('Unknown')
        
        # Convert player names to categorical codes
        raw_data['player_1_code'] = pd.Categorical(raw_data['player_1']).codes
        raw_data['player_2_code'] = pd.Categorical(raw_data['player_2']).codes

        # Drop unnecessary columns based on ATP tennis data format
        columns_to_drop = ['Odd_1', 'Odd_2']
        raw_data = raw_data.drop(columns=columns_to_drop)

        # Change the date column to a timestamp
        raw_data['Date'] = pd.to_datetime(raw_data['Date'])
        
        print(f"Dropped the following columns: {columns_to_drop}")
        
        return raw_data
        
        # ... existing code ...
    
    
    