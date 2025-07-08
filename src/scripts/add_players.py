import os
import pandas as pd
import sys

# Add the project root directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.append(project_root)

def main():
    # Define the path to the CSV file
    csv_path = os.path.join(project_root, 'data', 'atp_real.csv')
    
    # Read the existing CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded existing CSV with {len(df)} rows")
    except FileNotFoundError:
        # Create a new DataFrame if the file doesn't exist
        print("CSV file not found. Creating a new one.")
        df = pd.DataFrame(columns=[
            'Tournament', 'Date', 'Series', 'Court', 'Surface', 'Round', 
            'Best of', 'Player_1', 'Player_2', 'Rank_1', 'Rank_2', 
            'Pts_1', 'Pts_2', 'Odd_1', 'Odd_2', 'Score', 'Winner'
        ])
    
    # Define new matches from the image
    new_matches = [
        # 2nd Round Qualifying - Court 5
        {
            'Tournament': 'ATP Qualifying',
            'Date': '2023-06-15',  # Sample date
            'Series': 'Qualifying',
            'Court': 'Outdoor',
            'Surface': 'Hard',
            'Round': '2nd Round Qualifying',
            'Best of': 3,
            'Player_1': 'Brandon Holt',
            'Player_2': 'Gabriel Diallo',
            'Rank_1': 17,
            'Rank_2': 6,
            'Pts_1': 0,
            'Pts_2': 0,
            'Odd_1': 1.5,  # Sample odds
            'Odd_2': 2.5,  # Sample odds
            'Score': '6-4 2-6 7-5',
            'Winner': 'Brandon Holt'
        },
        
        # 2nd Round Qualifying - Court 3
        {
            'Tournament': 'ATP Qualifying',
            'Date': '2023-06-15',  # Sample date
            'Series': 'Qualifying',
            'Court': 'Outdoor',
            'Surface': 'Hard',
            'Round': '2nd Round Qualifying',
            'Best of': 3,
            'Player_1': 'Billy Harris',
            'Player_2': 'Hugo Gaston',
            'Rank_1': 13,
            'Rank_2': 7,
            'Pts_1': 0,
            'Pts_2': 0,
            'Odd_1': 1.8,  # Sample odds
            'Odd_2': 2.0,  # Sample odds
            'Score': '7-5 6-2',
            'Winner': 'Billy Harris'
        },
        
        # 2nd Round Qualifying - Court 6 (first match)
        {
            'Tournament': 'ATP Qualifying',
            'Date': '2023-06-15',  # Sample date
            'Series': 'Qualifying',
            'Court': 'Outdoor',
            'Surface': 'Hard',
            'Round': '2nd Round Qualifying',
            'Best of': 3,
            'Player_1': 'Tristan Schoolkate',
            'Player_2': 'Adam Walton',
            'Rank_1': 24,
            'Rank_2': 8,
            'Pts_1': 0,
            'Pts_2': 0,
            'Odd_1': 1.7,  # Sample odds
            'Odd_2': 2.2,  # Sample odds
            'Score': '6-3 6-4',
            'Winner': 'Tristan Schoolkate'
        },
        
        # 2nd Round Qualifying - Court 6 (second match)
        {
            'Tournament': 'ATP Qualifying',
            'Date': '2023-06-15',  # Sample date
            'Series': 'Qualifying',
            'Court': 'Outdoor',
            'Surface': 'Hard',
            'Round': '2nd Round Qualifying',
            'Best of': 3,
            'Player_1': 'Rei Sakamoto',
            'Player_2': 'James Duckworth',
            'Rank_1': 0,  # WC indicates wildcard, not an actual ranking
            'Rank_2': 9,
            'Pts_1': 0,
            'Pts_2': 0,
            'Odd_1': 2.0,  # Sample odds
            'Odd_2': 1.9,  # Sample odds
            'Score': '7-5 7-6(5)',
            'Winner': 'Rei Sakamoto'
        }
    ]
    
    # Add the new matches to the DataFrame
    df_new = pd.DataFrame(new_matches)
    df = pd.concat([df, df_new], ignore_index=True)
    
    # Save the updated DataFrame
    df.to_csv(csv_path, index=False)
    print(f"Added {len(new_matches)} matches to the CSV file")
    print(f"Total rows in CSV: {len(df)}")

if __name__ == "__main__":
    main() 