#!/usr/bin/env python3
import pandas as pd
import nfl_data_py as nfl

def download_nfl_data():
    # try downloading data year by year to see what's available
    print("checking available nfl data...")
    available_years = []
    all_data = []
    
    for year in range(2010, 2025):
        try:
            print(f"trying year {year}...")
            # try without specifying columns first to avoid game_id issues
            pbp = nfl.import_pbp_data([year])
            # then filter to needed columns if they exist
            needed_cols = ['season', 'week', 'posteam', 'epa', 'success', 'pass', 'air_yards', 'yards_gained', 'play_type']
            available_cols = [col for col in needed_cols if col in pbp.columns]
            pbp = pbp[available_cols]
            
            if len(pbp) > 0:
                available_years.append(year)
                all_data.append(pbp)
                print(f"  {year}: {len(pbp)} plays with columns {available_cols}")
            else:
                print(f"  {year}: no data")
        except Exception as e:
            print(f"  {year}: error - {e}")
    
    if not all_data:
        print("no data available")
        return
        
    print(f"available years: {available_years}")
    
    # combine all available data
    pbp = pd.concat(all_data, ignore_index=True)
    
    # filter to regular season only and clean data
    pbp = pbp[(pbp['week'] <= 18) & (pbp['week'] >= 1)]
    pbp = pbp[pbp['posteam'].notna() & (pbp['play_type'].isin(['pass', 'run']))]
    
    # calculate weekly team statistics
    weekly_stats = pbp.groupby(['season', 'week', 'posteam']).agg({
        'epa': 'mean',
        'success': 'mean', 
        'pass': 'mean',
        'air_yards': 'mean',
        'yards_gained': 'mean',
        'play_type': 'count'
    }).reset_index()
    
    # rename columns to match expected features
    weekly_stats.columns = ['season', 'week', 'team', 'off_epa_per_play', 'success_rate', 'pass_rate', 'avg_air_yards', 'yards_per_play', 'total_plays']
    
    # fill missing values with team averages
    weekly_stats = weekly_stats.fillna(weekly_stats.groupby('team').transform('mean'))
    
    # split training and testing data
    training_data = weekly_stats[weekly_stats['season'] <= 2023]
    testing_data = weekly_stats[weekly_stats['season'] == 2024]
    
    print(f"training seasons: {sorted(training_data['season'].unique())}")
    print(f"testing seasons: {sorted(testing_data['season'].unique())}")
    
    # append new training data to existing csv (if it exists)
    try:
        existing_training = pd.read_csv('nfl_training_data.csv')
        # combine and remove duplicates
        combined_training = pd.concat([existing_training, training_data]).drop_duplicates(
            subset=['season', 'week', 'team'], keep='last'
        )
        combined_training.to_csv('nfl_training_data.csv', index=False)
        print(f"updated training data: {len(combined_training)} total observations")
    except FileNotFoundError:
        training_data.to_csv('nfl_training_data.csv', index=False)
        print(f"created new training data: {len(training_data)} observations")
    
    # save testing data
    testing_data.to_csv('nfl_testing_data.csv', index=False)
    print(f"saved {len(testing_data)} testing observations")
    print("features: off_epa_per_play, success_rate, pass_rate, avg_air_yards, yards_per_play, total_plays")

if __name__ == "__main__":
    download_nfl_data()
