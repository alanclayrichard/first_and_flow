# Data Generation for First and Flow

This directory contains all scripts and utilities for generating NFL training data for the flow matching model.

## Directory Structure

```
data/
├── generate_data.R          # Main data generation pipeline
├── nfl_data_processor.R     # Core NFL data processing script
├── test_processor.R         # Data processing validation
├── install_requirements.R   # R package dependencies
├── usage_examples.R         # Usage documentation
├── raw/                     # Raw downloaded data (cached)
└── processed/               # Processed training data
    └── nfl_flow_training_data.csv  # Main training dataset
```

## Quick Start

```bash
# Install required R packages
Rscript install_requirements.R

# Test the data pipeline
Rscript test_processor.R

# Generate training data (2018-2023)
Rscript generate_data.R

# Generate custom date range
Rscript generate_data.R 2020 2023
```

## Data Pipeline

1. **Raw Data Download** (`nfl_data_processor.R`)
   - Downloads play-by-play data from nflfastR
   - Filters to relevant offensive plays
   - Caches raw data in `raw/` directory

2. **Feature Engineering** 
   - Aggregates team-level weekly metrics
   - Calculates flow-relevant statistics
   - Handles missing data and outliers

3. **Output Generation**
   - Saves processed data to `processed/nfl_flow_training_data.csv`
   - Validates data quality and completeness
   - Generates summary statistics

## Output Schema

The training dataset contains team-week level observations with:

| Column | Type | Description |
|--------|------|-------------|
| `season` | int | NFL season year |
| `week` | int | Week number (1-18 regular, 19+ playoffs) |
| `team` | str | Team abbreviation (e.g., "BUF", "KC") |
| `off_epa_per_play` | float | Offensive Expected Points Added per play |
| `success_rate` | float | Success rate (0-1 scale) |
| `pass_rate` | float | Pass percentage (0-1 scale) |
| `avg_air_yards` | float | Average air yards on passes |
| `total_plays` | int | Total offensive plays |

## Data Quality

- **Completeness**: Only includes teams/weeks with ≥10 plays
- **Validation**: Automatic range checking and outlier detection
- **Consistency**: Standardized team abbreviations and metrics
- **Python-ready**: No R factors, clean column names

## Flow Matching Integration

The processed data is optimized for flow matching model development:

- **Team performance vectors**: Weekly offensive metrics per team
- **Temporal structure**: Sequential weeks for flow modeling
- **Standardized features**: Normalized scales for model training
- **Rich context**: Multiple performance dimensions

## Troubleshooting

**Common Issues:**

1. **Package Installation**: Run `install_requirements.R` first
2. **Memory Issues**: Reduce year range for large datasets
3. **Network Issues**: Check internet connection for nflfastR downloads
4. **Data Validation**: Use `test_processor.R` to verify setup

**Performance Notes:**

- Single season: ~1-2 minutes
- 5+ seasons: ~5-15 minutes
- Memory usage: ~2-4GB for full pipeline
- Output size: ~50KB per season

## Next Steps

After data generation, proceed to flow matching model development in `src/first_and_flow/`.
