# NFL Data Processing - Example Usage
# 
# This file demonstrates how to use the nfl_data_processor.R script

# =============================================================================
# BASIC USAGE EXAMPLES
# =============================================================================

# Example 1: Use default parameters (2022-2023 seasons)
# Rscript nfl_data_processor.R

# Example 2: Specify custom year range
# Rscript nfl_data_processor.R 2020 2023

# Example 3: Specify custom year range and output filename
# Rscript nfl_data_processor.R 2019 2021 custom_nfl_data.csv

# Example 4: Single season
# Rscript nfl_data_processor.R 2023 2023 season_2023.csv

# =============================================================================
# EXPECTED OUTPUT STRUCTURE
# =============================================================================

# The output CSV will contain the following columns:
# - season: Year of the NFL season (integer)
# - week: Week number (1-18 for regular season, 19+ for playoffs)
# - team: Team abbreviation (e.g., "BUF", "KC", "SF")
# - off_epa_per_play: Offensive Expected Points Added per play (numeric)
# - success_rate: Percentage of successful plays (0-1 scale)
# - pass_rate: Percentage of plays that are passes (0-1 scale)
# - avg_air_yards: Average air yards on passing plays (numeric)
# - total_plays: Total number of plays in that team-week (integer)

# =============================================================================
# PYTHON INTEGRATION EXAMPLE
# =============================================================================

# After running the R script, you can easily load the data in Python:
# 
# import pandas as pd
# 
# # Load the processed NFL data
# nfl_data = pd.read_csv('nfl_weekly_team_stats.csv')
# 
# # Basic exploration
# print(nfl_data.head())
# print(nfl_data.info())
# print(nfl_data.describe())
# 
# # The data is ready for machine learning pipelines!

# =============================================================================
# PERFORMANCE NOTES
# =============================================================================

# Processing time varies by number of seasons:
# - Single season (~18 weeks): ~30-60 seconds
# - 2-3 seasons: ~1-3 minutes  
# - 5+ seasons: ~3-10 minutes
#
# Memory usage is generally modest (< 2GB RAM for 5 seasons)
# Output file size: ~50-100 KB per season
