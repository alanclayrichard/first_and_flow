#!/usr/bin/env Rscript

#' Quick Test Script for NFL Data Processor
#' 
#' This script runs a minimal test of the nfl_data_processor.R script
#' using a small dataset to verify functionality.

cat("=== NFL Data Processor Test ===\n")
cat("Testing with 2023 season, weeks 1-2 only\n\n")

# Check if required packages are available
required_packages <- c("nflfastR", "dplyr", "readr")
missing_packages <- c()

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    missing_packages <- c(missing_packages, pkg)
  }
}

if (length(missing_packages) > 0) {
  cat("ERROR: Missing required packages:", paste(missing_packages, collapse = ", "), "\n")
  cat("Please run: source('install_requirements.R')\n")
  quit(status = 1)
}

# Load required libraries
suppressPackageStartupMessages({
  library(nflfastR)
  library(dplyr)
  library(readr)
})

cat("All packages loaded successfully.\n")

# Test function: Download and process a small sample
test_processor <- function() {
  cat("Downloading sample data (2023, weeks 1-2)...\n")
  
  tryCatch({
    # Download just a small sample
    pbp_sample <- load_pbp(seasons = 2023) %>%
      filter(week <= 2, !is.na(epa), play_type %in% c("pass", "run"))
    
    cat("Downloaded", nrow(pbp_sample), "plays.\n")
    
    # Process sample data
    test_stats <- pbp_sample %>%
      group_by(season, week, posteam) %>%
      summarise(
        off_epa_per_play = round(mean(epa, na.rm = TRUE), 4),
        success_rate = round(mean(success, na.rm = TRUE), 4),
        pass_rate = round(mean(play_type == "pass", na.rm = TRUE), 4),
        avg_air_yards = round(mean(air_yards[play_type == "pass"], na.rm = TRUE), 4),
        total_plays = n(),
        .groups = "drop"
      ) %>%
      mutate(
        avg_air_yards = ifelse(is.nan(avg_air_yards), 0, avg_air_yards)
      ) %>%
      rename(team = posteam) %>%
      filter(!is.na(team), total_plays >= 5)
    
    cat("Processed", nrow(test_stats), "team-week observations.\n")
    
    # Validate basic structure
    expected_cols <- c("season", "week", "team", "off_epa_per_play", 
                       "success_rate", "pass_rate", "avg_air_yards", "total_plays")
    
    if (all(expected_cols %in% names(test_stats))) {
      cat("✓ All expected columns present\n")
    } else {
      stop("✗ Missing expected columns")
    }
    
    # Check data ranges
    if (all(test_stats$success_rate >= 0 & test_stats$success_rate <= 1, na.rm = TRUE)) {
      cat("✓ Success rates in valid range\n")
    } else {
      warning("✗ Some success rates outside [0,1] range")
    }
    
    if (all(test_stats$pass_rate >= 0 & test_stats$pass_rate <= 1, na.rm = TRUE)) {
      cat("✓ Pass rates in valid range\n")
    } else {
      warning("✗ Some pass rates outside [0,1] range")
    }
    
    # Save test output
    test_filename <- "test_nfl_output.csv"
    write_csv(test_stats, test_filename)
    cat("✓ Test data saved to", test_filename, "\n")
    
    # Display sample
    cat("\nSample output:\n")
    print(head(test_stats, 5))
    
    return(TRUE)
    
  }, error = function(e) {
    cat("✗ Test failed:", e$message, "\n")
    return(FALSE)
  })
}

# Run the test
test_result <- test_processor()

if (test_result) {
  cat("\n=== TEST PASSED ===\n")
  cat("The NFL data processor appears to be working correctly.\n")
  cat("You can now run the full script with:\n")
  cat("Rscript nfl_data_processor.R [start_year] [end_year] [output_file]\n")
} else {
  cat("\n=== TEST FAILED ===\n")
  cat("Please check the error messages above and ensure all requirements are met.\n")
  quit(status = 1)
}

cat("\nTest completed.\n")
