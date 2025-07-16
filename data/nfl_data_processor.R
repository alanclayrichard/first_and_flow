#!/usr/bin/env Rscript

#' NFL Play-by-Play Data Processing Script
#' 
#' This script downloads NFL play-by-play data using nflfastR and aggregates
#' team-level weekly performance metrics for machine learning pipelines.
#' 
#' Usage: Rscript nfl_data_processor.R [start_year] [end_year] [output_file]
#' 
#' Arguments:
#'   start_year: First season to include (default: 2022)
#'   end_year: Last season to include (default: 2023)
#'   output_file: Output CSV filename (default: nfl_weekly_team_stats.csv)
#' 
#' Example: Rscript nfl_data_processor.R 2020 2023 my_nfl_data.csv

# =============================================================================
# PACKAGE INSTALLATION AND LOADING
# =============================================================================

#' Function to install packages if they don't exist
install_if_missing <- function(packages) {
  for (pkg in packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
      cat("Installing package:", pkg, "\n")
      install.packages(pkg, dependencies = TRUE, repos = "https://cran.r-project.org")
      library(pkg, character.only = TRUE)
    }
  }
}

# Required packages
required_packages <- c("nflfastR", "dplyr", "readr", "stringr")

cat("Checking and installing required packages...\n")
install_if_missing(required_packages)

# Load libraries
suppressPackageStartupMessages({
  library(nflfastR)
  library(dplyr)
  library(readr)
  library(stringr)
})

cat("All packages loaded successfully.\n\n")

# =============================================================================
# COMMAND LINE ARGUMENT PARSING
# =============================================================================

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Set default values
start_year <- 2022
end_year <- 2023
output_file <- "nfl_weekly_team_stats.csv"

# Parse command line arguments
if (length(args) >= 1 && !is.na(as.numeric(args[1]))) {
  start_year <- as.numeric(args[1])
}

if (length(args) >= 2 && !is.na(as.numeric(args[2]))) {
  end_year <- as.numeric(args[2])
}

if (length(args) >= 3 && nchar(args[3]) > 0) {
  output_file <- args[3]
}

# Validate year range
if (start_year > end_year) {
  stop("Error: start_year must be less than or equal to end_year")
}

if (start_year < 1999) {
  stop("Error: nflfastR data is only available from 1999 onwards")
}

current_year <- as.numeric(format(Sys.Date(), "%Y"))
if (end_year > current_year) {
  warning(paste("Warning: end_year", end_year, "is in the future. Using", current_year, "instead."))
  end_year <- current_year
}

cat("Processing NFL data for seasons:", start_year, "to", end_year, "\n")
cat("Output file:", output_file, "\n\n")

# =============================================================================
# DATA DOWNLOAD AND PROCESSING
# =============================================================================

#' Function to calculate team weekly aggregated statistics
calculate_team_weekly_stats <- function(pbp_data) {
  cat("Calculating team-level weekly statistics...\n")
  
  # Filter to relevant plays (exclude penalties, timeouts, etc.)
  relevant_plays <- pbp_data %>%
    filter(
      !is.na(epa),                    # Must have EPA data
      !is.na(success),                # Must have success data
      play_type %in% c("pass", "run"), # Only pass and run plays
      !is.na(posteam)                 # Must have possessing team
    )
  
  # Calculate offensive statistics by team and week
  offensive_stats <- relevant_plays %>%
    group_by(season, week, posteam) %>%
    summarise(
      # Offensive EPA per play
      off_epa_per_play = round(mean(epa, na.rm = TRUE), 4),
      
      # Success rate (percentage of successful plays)
      success_rate = round(mean(success, na.rm = TRUE), 4),
      
      # Pass rate (percentage of plays that are passes)
      pass_rate = round(mean(play_type == "pass", na.rm = TRUE), 4),
      
      # Average air yards (only for pass plays)
      avg_air_yards = round(mean(air_yards[play_type == "pass"], na.rm = TRUE), 4),
      
      # Total plays
      total_plays = n(),
      
      .groups = "drop"
    ) %>%
    # Handle NaN values for teams with no passing plays
    mutate(
      avg_air_yards = ifelse(is.nan(avg_air_yards), 0, avg_air_yards)
    ) %>%
    # Rename team column for clarity
    rename(team = posteam)
  
  return(offensive_stats)
}

#' Main data processing function
process_nfl_data <- function(start_year, end_year) {
  cat("Downloading NFL play-by-play data...\n")
  
  # Create vector of seasons to download
  seasons <- start_year:end_year
  
  # Download play-by-play data for all seasons
  # nflfastR automatically handles multiple seasons
  tryCatch({
    pbp_data <- load_pbp(seasons = seasons)
    cat("Successfully downloaded", nrow(pbp_data), "plays from", length(seasons), "seasons.\n\n")
  }, error = function(e) {
    stop(paste("Error downloading data:", e$message))
  })
  
  # Process the data to get team weekly stats
  team_stats <- calculate_team_weekly_stats(pbp_data)
  
  # Filter out rows with missing critical data
  team_stats <- team_stats %>%
    filter(
      !is.na(off_epa_per_play),
      !is.na(success_rate),
      !is.na(pass_rate),
      total_plays >= 10  # Ensure minimum sample size
    ) %>%
    # Sort by season, week, and team for consistency
    arrange(season, week, team)
  
  cat("Processed data contains", nrow(team_stats), "team-week observations.\n")
  
  return(team_stats)
}

# =============================================================================
# DATA VALIDATION AND EXPORT
# =============================================================================

#' Function to validate the processed data
validate_data <- function(data) {
  cat("Validating processed data...\n")
  
  # Check for required columns
  required_cols <- c("season", "week", "team", "off_epa_per_play", 
                     "success_rate", "pass_rate", "avg_air_yards", "total_plays")
  
  missing_cols <- setdiff(required_cols, names(data))
  if (length(missing_cols) > 0) {
    stop(paste("Missing required columns:", paste(missing_cols, collapse = ", ")))
  }
  
  # Check data types and ranges
  if (any(data$success_rate < 0 | data$success_rate > 1, na.rm = TRUE)) {
    warning("Success rate values outside [0, 1] range detected")
  }
  
  if (any(data$pass_rate < 0 | data$pass_rate > 1, na.rm = TRUE)) {
    warning("Pass rate values outside [0, 1] range detected")
  }
  
  if (any(data$total_plays <= 0, na.rm = TRUE)) {
    warning("Non-positive total_plays values detected")
  }
  
  # Summary statistics
  cat("Data validation summary:\n")
  cat("  Seasons:", min(data$season), "to", max(data$season), "\n")
  cat("  Weeks:", min(data$week), "to", max(data$week), "\n")
  cat("  Teams:", length(unique(data$team)), "unique teams\n")
  cat("  Total observations:", nrow(data), "\n")
  cat("  Average plays per team-week:", round(mean(data$total_plays), 1), "\n\n")
  
  return(TRUE)
}

#' Function to save data with Python-friendly formatting
save_clean_csv <- function(data, filename) {
  cat("Saving data to CSV file...\n")
  
  # Ensure all character columns are properly formatted
  # Convert any factors to characters (Python-friendly)
  data_clean <- data %>%
    mutate(
      across(where(is.factor), as.character),
      # Ensure numeric columns have consistent precision
      off_epa_per_play = round(off_epa_per_play, 4),
      success_rate = round(success_rate, 4),
      pass_rate = round(pass_rate, 4),
      avg_air_yards = round(avg_air_yards, 4)
    )
  
  # Write CSV with settings optimized for Python reading
  tryCatch({
    write_csv(data_clean, filename, na = "")
    cat("Successfully saved", nrow(data_clean), "rows to", filename, "\n")
  }, error = function(e) {
    stop(paste("Error saving file:", e$message))
  })
  
  return(invisible(TRUE))
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Execute main processing pipeline
cat("=== NFL Data Processing Pipeline ===\n")
cat("Start time:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")

tryCatch({
  # Process the data
  team_weekly_stats <- process_nfl_data(start_year, end_year)
  
  # Validate the results
  validate_data(team_weekly_stats)
  
  # Save to CSV
  save_clean_csv(team_weekly_stats, output_file)
  
  # Final summary
  cat("\n=== Processing Complete ===\n")
  cat("Data successfully processed and saved to:", output_file, "\n")
  cat("End time:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
  
  # Display sample of the data
  cat("\nSample of processed data:\n")
  print(head(team_weekly_stats, 10))
  
}, error = function(e) {
  cat("ERROR:", e$message, "\n")
  quit(status = 1)
})

cat("\nScript completed successfully!\n")
