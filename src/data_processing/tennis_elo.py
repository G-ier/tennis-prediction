import pandas as pd
import numpy as np
from collections import defaultdict
import os

# Read the data
def load_data(file_path, row_limit=None):
    # Read the CSV file with proper handling for the format issues
    matches_raw = pd.read_csv(file_path, sep=',', engine='python', on_bad_lines='skip', nrows=row_limit)
    
    # Map Series to tourney_level (use 'G' for Grand Slam, others as is)
    matches_raw['tourney_level'] = matches_raw['Series'].apply(
        lambda x: 'G' if 'Grand Slam' in str(x) else x
    )
    
    # Include Surface for court-specific ELO
    matches_raw['surface'] = matches_raw['Surface']
    
    # Create match_num as a sequential number for each date
    matches_raw = matches_raw.sort_values('Date')
    matches_raw['match_num'] = matches_raw.groupby('Date').cumcount() + 1
    
    # Select required columns
    matches = matches_raw[['Player_1', 'Player_2', 'Winner', 'tourney_level', 'Date', 'match_num', 'surface']]
    
    # Rename Date to tourney_date
    matches = matches.rename(columns={'Date': 'tourney_date'})
    
    # Convert date format
    matches['tourney_date'] = pd.to_datetime(matches['tourney_date'], format='%Y-%m-%d')
    
    # Sort by date and match number
    matches = matches.sort_values(by=['tourney_date', 'match_num']).reset_index(drop=True)
    
    return matches

# Initialize data structures to store Elo ratings and match counts
def initialize_structures():
    # Overall ELO
    players_to_elo = {}
    matches_count = defaultdict(int)
    
    # Court-specific ELO
    court_elo = {
        'Hard': {},
        'Clay': {},
        'Grass': {},
        'Carpet': {},
        'Overall': {}  # Keep overall ELO as well
    }
    
    court_matches_count = {
        'Hard': defaultdict(int),
        'Clay': defaultdict(int),
        'Grass': defaultdict(int),
        'Carpet': defaultdict(int),
        'Overall': defaultdict(int)
    }
    
    return players_to_elo, matches_count, court_elo, court_matches_count

# Update match count for both players
def update_matches_count(player_a, player_b, matches_count, court_matches_count, surface):
    # Update overall match counts
    matches_count[player_a] += 1
    matches_count[player_b] += 1
    
    # Update court-specific match counts
    court_matches_count['Overall'][player_a] += 1
    court_matches_count['Overall'][player_b] += 1
    
    if surface in court_matches_count:
        court_matches_count[surface][player_a] += 1
        court_matches_count[surface][player_b] += 1
    
    return matches_count, court_matches_count

# Update Elo ratings after a match
def update_elo(players_to_elo, player_a, player_b, winner, level, match_date, match_num, matches_count):
    first_date = pd.Timestamp('1900-01-01')
    
    # Get current ratings, or initialize if not present
    if player_a not in players_to_elo:
        players_to_elo[player_a] = pd.DataFrame({'ranking': [1500], 'date': [first_date], 'num': [0]})
    if player_b not in players_to_elo:
        players_to_elo[player_b] = pd.DataFrame({'ranking': [1500], 'date': [first_date], 'num': [0]})
    
    r_a = players_to_elo[player_a]['ranking'].iloc[-1]
    r_b = players_to_elo[player_b]['ranking'].iloc[-1]
    
    # Calculate expected scores
    e_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    e_b = 1 / (1 + 10 ** ((r_a - r_b) / 400))
    
    # Set actual scores
    if winner == player_a:
        s_a, s_b = 1, 0
    else:
        s_a, s_b = 0, 1
    
    # Calculate K-factors
    k_a = 250 / ((matches_count[player_a] + 5) ** 0.4)
    k_b = 250 / ((matches_count[player_b] + 5) ** 0.4)
    
    # Adjust K-factor for Grand Slam tournaments (G)
    k = 1.1 if level == 'G' else 1
    
    # Calculate new ratings
    r_a_new = r_a + (k * k_a) * (s_a - e_a)
    r_b_new = r_b + (k * k_b) * (s_b - e_b)
    
    # Update player dataframes
    players_to_elo[player_a] = pd.concat([
        players_to_elo[player_a],
        pd.DataFrame({'ranking': [r_a_new], 'date': [match_date], 'num': [match_num]})
    ]).reset_index(drop=True)
    
    players_to_elo[player_b] = pd.concat([
        players_to_elo[player_b],
        pd.DataFrame({'ranking': [r_b_new], 'date': [match_date], 'num': [match_num]})
    ]).reset_index(drop=True)
    
    return players_to_elo

# Update court-specific Elo ratings
def update_court_elo(court_elo, player_a, player_b, winner, level, match_date, match_num, court_matches_count, surface):
    first_date = pd.Timestamp('1900-01-01')
    
    # Update for overall court type
    surface_types = ['Overall']
    
    # Add specific surface if valid
    if surface in court_elo:
        surface_types.append(surface)
    
    # Update ELO for each applicable surface type
    for surf in surface_types:
        if player_a not in court_elo[surf]:
            court_elo[surf][player_a] = pd.DataFrame({'ranking': [1500], 'date': [first_date], 'num': [0]})
        if player_b not in court_elo[surf]:
            court_elo[surf][player_b] = pd.DataFrame({'ranking': [1500], 'date': [first_date], 'num': [0]})
        
        r_a = court_elo[surf][player_a]['ranking'].iloc[-1]
        r_b = court_elo[surf][player_b]['ranking'].iloc[-1]
        
        # Calculate expected scores
        e_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
        e_b = 1 / (1 + 10 ** ((r_a - r_b) / 400))
        
        # Set actual scores
        if winner == player_a:
            s_a, s_b = 1, 0
        else:
            s_a, s_b = 0, 1
        
        # Calculate K-factors (using court-specific match counts)
        k_a = 250 / ((court_matches_count[surf][player_a] + 5) ** 0.4)
        k_b = 250 / ((court_matches_count[surf][player_b] + 5) ** 0.4)
        
        # Adjust K-factor for Grand Slam tournaments (G)
        k = 1.1 if level == 'G' else 1
        
        # Calculate new ratings
        r_a_new = r_a + (k * k_a) * (s_a - e_a)
        r_b_new = r_b + (k * k_b) * (s_b - e_b)
        
        # Update player dataframes
        court_elo[surf][player_a] = pd.concat([
            court_elo[surf][player_a],
            pd.DataFrame({'ranking': [r_a_new], 'date': [match_date], 'num': [match_num]})
        ]).reset_index(drop=True)
        
        court_elo[surf][player_b] = pd.concat([
            court_elo[surf][player_b],
            pd.DataFrame({'ranking': [r_b_new], 'date': [match_date], 'num': [match_num]})
        ]).reset_index(drop=True)
    
    return court_elo

# Process all matches to compute Elo ratings
def compute_elo(matches):
    players_to_elo, matches_count, court_elo, court_matches_count = initialize_structures()
    
    # Process each match
    for index, row in matches.iterrows():
        player_a = row['Player_1']
        player_b = row['Player_2']
        winner = row['Winner']
        level = row['tourney_level']
        match_date = row['tourney_date']
        match_num = row['match_num']
        surface = row['surface']
        
        # Update match counts
        matches_count, court_matches_count = update_matches_count(
            player_a, player_b, matches_count, court_matches_count, surface
        )
        
        # Update overall Elo ratings
        players_to_elo = update_elo(
            players_to_elo, player_a, player_b, winner, level, 
            match_date, match_num, matches_count
        )
        
        # Update court-specific Elo ratings
        court_elo = update_court_elo(
            court_elo, player_a, player_b, winner, level,
            match_date, match_num, court_matches_count, surface
        )
    
    return players_to_elo, court_elo

# Get players with highest Elo ratings
def summary_players(players_to_elo):
    players_to_max = pd.DataFrame({'ranking': [1500], 'meanr': [1500], 'medianr': [1500], 'name': ['Nobody']})
    
    for player, data in players_to_elo.items():
        new_row = pd.DataFrame({
            'ranking': [data['ranking'].max()],
            'meanr': [data['ranking'].mean()],
            'medianr': [data['ranking'].median()],
            'name': [player]
        })
        players_to_max = pd.concat([players_to_max, new_row], ignore_index=True)
    
    players_to_max = players_to_max.sort_values(by='ranking', ascending=False).reset_index(drop=True)
    return players_to_max

# Calculate court-specific ELO summary
def court_elo_summary(court_elo):
    surface_summaries = {}
    
    for surface, players in court_elo.items():
        if not players:  # Skip if no data for this surface
            continue
            
        players_to_max = pd.DataFrame({'ranking': [1500], 'name': ['Nobody']})
        
        for player, data in players.items():
            if len(data) <= 1:  # Skip players with only initial rating
                continue
                
            new_row = pd.DataFrame({
                'ranking': [data['ranking'].max()],
                'name': [player]
            })
            players_to_max = pd.concat([players_to_max, new_row], ignore_index=True)
        
        players_to_max = players_to_max.sort_values(by='ranking', ascending=False).reset_index(drop=True)
        surface_summaries[surface] = players_to_max
    
    return surface_summaries

# Calculate current and peak ELO ratings for all players
def current_and_peak_elo(players_to_elo):
    player_stats = pd.DataFrame(columns=['name', 'current_elo', 'peak_elo', 'peak_date'])
    
    for player, data in players_to_elo.items():
        if len(data) <= 1:  # Skip players with only initial rating
            continue
            
        current_elo = data['ranking'].iloc[-1]
        peak_elo = data['ranking'].max()
        peak_idx = data['ranking'].idxmax()
        peak_date = data['date'].iloc[peak_idx]
        
        new_row = pd.DataFrame({
            'name': [player],
            'current_elo': [round(current_elo, 2)],
            'peak_elo': [round(peak_elo, 2)],
            'peak_date': [peak_date]
        })
        
        player_stats = pd.concat([player_stats, new_row], ignore_index=True)
    
    # Sort by peak ELO in descending order
    player_stats = player_stats.sort_values(by='peak_elo', ascending=False).reset_index(drop=True)
    
    return player_stats

def get_recent_performance(matches_df, row_limit=6735, recent_matches=15):
    """
    Calculate the performance (win/loss record) in the last N matches for each player 
    up to the specified row limit. Creates a separate DataFrame for player statistics.
    
    Args:
        matches_df (pd.DataFrame): DataFrame containing processed match data with Player_1, Player_2, and Winner
        row_limit (int): Row limit to consider (default: 6735)
        recent_matches (int): Number of recent matches to consider (default: 15)
    
    Returns:
        tuple: (Original DataFrame, Player statistics DataFrame)
    """
    # Create a working copy of matches up to the specified row limit
    matches_subset = matches_df.iloc[:row_limit].copy()
    
    # Initialize dictionary to store player's recent matches
    player_recent_matches = {}
    
    # Track win/loss for each player
    for _, match in matches_subset.iterrows():
        player_1 = match['Player_1']
        player_2 = match['Player_2']
        winner = match['Winner']
        
        # Initialize player records if they don't exist
        if player_1 not in player_recent_matches:
            player_recent_matches[player_1] = []
        if player_2 not in player_recent_matches:
            player_recent_matches[player_2] = []
        
        # Add win/loss records based on Winner column
        player_recent_matches[player_1].append(1 if winner == player_1 else 0)
        player_recent_matches[player_2].append(1 if winner == player_2 else 0)
        
        # Keep only the most recent N matches
        player_recent_matches[player_1] = player_recent_matches[player_1][-recent_matches:]
        player_recent_matches[player_2] = player_recent_matches[player_2][-recent_matches:]
    
    # Create a DataFrame for player statistics
    player_stats = []
    for player, results in player_recent_matches.items():
        total_matches = len(results)
        wins = sum(results)
        losses = total_matches - wins
        win_percentage = wins / total_matches if total_matches > 0 else 0
        
        player_stats.append({
            'player': player,
            'matches_played': total_matches,
            'wins': wins,
            'losses': losses,
            'win_percentage': win_percentage,
            'recent_form': results  # List of recent results (1=win, 0=loss)
        })
    
    # Convert player statistics to DataFrame
    player_stats_df = pd.DataFrame(player_stats)
    
    # Sort by win percentage in descending order
    player_stats_df = player_stats_df.sort_values(by='win_percentage', ascending=False).reset_index(drop=True)
    
    return matches_subset, player_stats_df

# Calculate court-specific current and peak ELO
def court_specific_elo(court_elo):
    surface_stats = {}
    
    for surface, players in court_elo.items():
        if not players:  # Skip if no data for this surface
            continue
            
        player_stats = pd.DataFrame(columns=['name', 'current_elo', 'peak_elo', 'peak_date'])
        
        for player, data in players.items():
            if len(data) <= 1:  # Skip players with only initial rating
                continue
                
            current_elo = data['ranking'].iloc[-1]
            peak_elo = data['ranking'].max()
            peak_idx = data['ranking'].idxmax()
            peak_date = data['date'].iloc[peak_idx]
            
            new_row = pd.DataFrame({
                'name': [player],
                'current_elo': [round(current_elo, 2)],
                'peak_elo': [round(peak_elo, 2)],
                'peak_date': [peak_date]
            })
            
            player_stats = pd.concat([player_stats, new_row], ignore_index=True)
        
        # Sort by peak ELO in descending order
        player_stats = player_stats.sort_values(by='peak_elo', ascending=False).reset_index(drop=True)
        surface_stats[surface] = player_stats
    
    return surface_stats

# Find peak rankings between two dates
def between_dates(players_to_elo, date1, date2):
    players_to_max = pd.DataFrame({'ranking': [1500], 'meanr': [1500], 'medianr': [1500], 'name': ['Nobody']})
    
    for player, data in players_to_elo.items():
        filtered_data = data[(data['date'] >= date1) & (data['date'] <= date2)]
        
        if not filtered_data.empty:
            new_row = pd.DataFrame({
                'ranking': [filtered_data['ranking'].max()],
                'meanr': [filtered_data['ranking'].mean()],
                'medianr': [filtered_data['ranking'].median()],
                'name': [player]
            })
            players_to_max = pd.concat([players_to_max, new_row], ignore_index=True)
    
    players_to_max = players_to_max.sort_values(by='ranking', ascending=False).reset_index(drop=True)
    return players_to_max

# Helper for date creation
def get_year_month(year, month):
    return pd.Timestamp(f"{year}-{month}-01")

"""
# Main execution
def main():
    # Load data with row limit of 56361
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../..'))
    elo_data_path = os.path.join(project_root, 'data', 'atp_short.csv')
    matches = load_data(elo_data_path, row_limit=1000)
    print(f"Loaded {len(matches)} matches for ELO calculation")
    
    # Get the elo ratings
    players_to_elo, court_elo = compute_elo(matches)

    print(players_to_elo)

    print(court_elo)
    
    # Get recent performance with a custom number of recent matches
    matches_df, player_stats_df = get_recent_performance(matches, recent_matches=15)
    
    # Display player statistics
    print("\nPlayer statistics (last 10 matches):")
    print(player_stats_df[['player', 'matches_played', 'wins', 'losses', 'win_percentage']].head(10))
    
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main() 
"""
