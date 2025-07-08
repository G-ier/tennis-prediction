import pandas as pd
import numpy as np
from collections import defaultdict

# Read the data
def load_data(file_path, row_limit=None):
    # Read the CSV file with proper handling for the format issues
    matches_raw = pd.read_csv(file_path, sep=',', engine='python', on_bad_lines='skip', nrows=row_limit)
    
    # Create winner_name and loser_name columns based on Player_1, Player_2, and Winner
    matches_raw['winner_name'] = matches_raw.apply(
        lambda row: row['Player_1'] if row['Winner'] == row['Player_1'] else row['Player_2'], 
        axis=1
    )
    matches_raw['loser_name'] = matches_raw.apply(
        lambda row: row['Player_2'] if row['Winner'] == row['Player_1'] else row['Player_1'], 
        axis=1
    )
    
    # Map Series to tourney_level (use 'G' for Grand Slam, others as is)
    matches_raw['tourney_level'] = matches_raw['Series'].apply(
        lambda x: 'G' if 'Grand Slam' in str(x) else x
    )
    
    # Include Surface for court-specific ELO
    matches_raw['surface'] = matches_raw['Surface']
    
    # Create match_num as a sequential number for each date
    matches_raw = matches_raw.sort_values('Date')
    matches_raw['match_num'] = matches_raw.groupby('Date').cumcount() + 1
    
    # Select required columns, including Player_1 and Player_2 for player identification
    matches = matches_raw[['Player_1', 'Player_2', 'winner_name', 'loser_name', 'tourney_level', 'Date', 'match_num', 'surface']]
    
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
        player_a = row['winner_name']
        player_b = row['loser_name']
        winner = row['winner_name']
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

# Main execution
def main():
    # Load data with row limit of 6735
    matches = load_data('data/atp_short.csv', row_limit=6735)
    
    # Compute ELO ratings
    players_to_elo, court_elo = compute_elo(matches)
    
    # Print top players by overall ELO
    print("\nTop Players by ELO Rating:")
    players_max = summary_players(players_to_elo)
    print(players_max.head(10))
    
    # Get player performance in the last 15 matches
    results_df, player_performance = get_recent_performance(matches, row_limit=6735)
    
    # Find and print the performance of top 5 players by ELO
    print("\nRecent Performance of Top 5 Players:")
    for i in range(1, 6):  # Top 5 players
        player_name = players_max.iloc[i]['name']
        if player_name in player_performance:
            perf = player_performance[player_name]
            print(f"{player_name}: {perf['wins']}-{perf['losses']} ({perf['win_percentage']:.2%})")
            
            # Show recent form as W/L sequence (most recent last)
            form_str = ''.join(['W' if r == 1 else 'L' for r in perf['recent_form']])
            print(f"Form: {form_str}")
            print()
    
    # Display the last 10 matches with win percentage columns
    print("\nLast 10 matches with win percentages for last 15 games:")
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', 120)  # Set width to avoid wrapping
    print(results_df.iloc[-10:][['Player_1', 'Player_2', 'player_1_win_pct_last15', 'player_2_win_pct_last15']])
    
    print("\nAnalysis complete.")

def get_recent_performance(matches_df, row_limit=6735):
    """
    Calculate the performance (win/loss record) in the last 15 matches for each player 
    up to the specified row limit. Adds win percentage columns for player_1 and player_2.
    
    Args:
        matches_df (pd.DataFrame): DataFrame containing match data
        row_limit (int): Row limit to consider (default: 6735)
    
    Returns:
        tuple: (Updated DataFrame with win percentages, Player performance dictionary)
    """
    # Create a working copy of matches up to the specified row limit
    matches_subset = matches_df.iloc[:row_limit].copy()
    
    # Sort matches by date to ensure chronological order
    matches_subset = matches_subset.sort_values(by=['tourney_date', 'match_num'])
    
    # Initialize dictionary to store player's recent matches
    player_recent_matches = {}
    
    # Track win/loss for each player
    for _, match in matches_subset.iterrows():
        winner = match['winner_name']
        loser = match['loser_name']
        
        # Initialize player records if they don't exist
        if winner not in player_recent_matches:
            player_recent_matches[winner] = []
        if loser not in player_recent_matches:
            player_recent_matches[loser] = []
        
        # Add win record for winner (1 = win)
        player_recent_matches[winner].append(1)
        
        # Add loss record for loser (0 = loss)
        player_recent_matches[loser].append(0)
        
        # Keep only the most recent 15 matches
        player_recent_matches[winner] = player_recent_matches[winner][-15:]
        player_recent_matches[loser] = player_recent_matches[loser][-15:]
    
    # Calculate performance statistics
    player_performance = {}
    for player, results in player_recent_matches.items():
        total_matches = len(results)
        wins = sum(results)
        losses = total_matches - wins
        win_percentage = wins / total_matches if total_matches > 0 else 0
        
        player_performance[player] = {
            'matches_played': total_matches,
            'wins': wins,
            'losses': losses,
            'win_percentage': win_percentage,
            'recent_form': results  # List of recent results (1=win, 0=loss)
        }
    
    # Add win percentage columns to the original dataframe for player_1 and player_2
    results_df = matches_subset.copy()
    
    # Initialize new columns with NaN
    results_df['player_1_win_pct_last15'] = np.nan
    results_df['player_2_win_pct_last15'] = np.nan
    
    # Fill in win percentages for each match based on player_1 and player_2
    for idx, row in results_df.iterrows():
        player_1 = row['Player_1']
        player_2 = row['Player_2']
        
        if player_1 in player_performance:
            results_df.at[idx, 'player_1_win_pct_last15'] = player_performance[player_1]['win_percentage']
            
        if player_2 in player_performance:
            results_df.at[idx, 'player_2_win_pct_last15'] = player_performance[player_2]['win_percentage']
    
    return results_df, player_performance

if __name__ == "__main__":
    main() 