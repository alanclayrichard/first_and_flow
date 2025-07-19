import numpy as np
import pandas as pd

def prepare_flow_data():
    # load data
    df = pd.read_csv('nfl_training_data.csv')
    
    # create team mapping (1-32)
    teams = sorted(df['team'].unique())
    team_map = {team: i+1 for i, team in enumerate(teams)}
    
    # get seasons and max weeks
    seasons = sorted(df['season'].unique())
    max_week = int(df['week'].max())
    
    # create start and end positions for each team trajectory
    start_positions = []
    end_positions = []
    
    for season in seasons:
        for team in teams:
            # filter data for this team/season
            team_data = df[(df['season'] == season) & (df['team'] == team)].sort_values('week')
            
            if len(team_data) < 2:  # need at least 2 weeks for start/end
                continue
            
            team_id = team_map[team] / 32  # normalized team id
            season_norm = (season - 2010) / 5  # normalized season
            
            # start position (early weeks 1-6)
            early_weeks = team_data[team_data['week'] <= 6]
            start_pos = [
                team_id,
                season_norm,
                early_weeks['off_epa_per_play'].mean(),
                early_weeks['success_rate'].mean(),
                early_weeks['pass_rate'].mean(),
                early_weeks['avg_air_yards'].mean() / 20,
                early_weeks['yards_per_play'].mean() / 10
            ]
            
            # end position (late weeks 13+)
            late_weeks = team_data[team_data['week'] >= 13]
            if len(late_weeks) > 0:
                end_pos = [
                    team_id,  # team id stays constant
                    season_norm,  # season stays constant
                    late_weeks['off_epa_per_play'].mean(),
                    late_weeks['success_rate'].mean(),
                    late_weeks['pass_rate'].mean(),
                    late_weeks['avg_air_yards'].mean() / 20,
                    late_weeks['yards_per_play'].mean() / 10
                ]
            else:
                # if no late weeks, use overall season average as end
                end_pos = [
                    team_id,
                    season_norm,
                    team_data['off_epa_per_play'].mean(),
                    team_data['success_rate'].mean(),
                    team_data['pass_rate'].mean(),
                    team_data['avg_air_yards'].mean() / 20,
                    team_data['yards_per_play'].mean() / 10
                ]
            
            start_positions.append(start_pos)
            end_positions.append(end_pos)
    
    return np.array(start_positions), np.array(end_positions), team_map

def prepare_test_data(test_file='nfl_testing_data.csv'):
    # load test data
    df = pd.read_csv(test_file)
    
    # create same team mapping as training
    teams = sorted(df['team'].unique())
    team_map = {team: i+1 for i, team in enumerate(teams)}
    
    # get early season data (weeks 1-6) as starting positions
    test_starts = []
    test_ends = []
    
    for team in teams:
        team_data = df[df['team'] == team].sort_values('week')
        
        if len(team_data) < 2:
            continue
            
        team_id = team_map[team] / 32
        season_norm = (2024 - 2010) / 5  # normalized test season
        
        # early weeks for starting position
        early_weeks = team_data[team_data['week'] <= 6]
        start_pos = [
            team_id,
            season_norm,
            early_weeks['off_epa_per_play'].mean(),
            early_weeks['success_rate'].mean(),
            early_weeks['pass_rate'].mean(),
            early_weeks['avg_air_yards'].mean() / 20,
            early_weeks['yards_per_play'].mean() / 10
        ]
        
        # late weeks for ground truth
        late_weeks = team_data[team_data['week'] >= 13]
        if len(late_weeks) > 0:
            end_pos = [
                team_id,
                season_norm,
                late_weeks['off_epa_per_play'].mean(),
                late_weeks['success_rate'].mean(),
                late_weeks['pass_rate'].mean(),
                late_weeks['avg_air_yards'].mean() / 20,
                late_weeks['yards_per_play'].mean() / 10
            ]
        else:
            end_pos = start_pos.copy()  # fallback
            
        test_starts.append(start_pos)
        test_ends.append(end_pos)
    
    return np.array(test_starts), np.array(test_ends), team_map

if __name__ == "__main__":
    p, q, team_mapping = prepare_flow_data()
    print(f"start positions (p) shape: {p.shape}")
    print(f"end positions (q) shape: {q.shape}")
    print(f"sample start: {p[0]}")
    print(f"sample end: {q[0]}")