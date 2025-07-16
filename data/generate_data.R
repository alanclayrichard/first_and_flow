#!/usr/bin/env Rscript

#' NFL Data Generation Pipeline
#' 
#' This script coordinates the entire data generation process for the 
#' first_and_flow project. It downloads, processes, and validates NFL data
#' for use in flow matching models.
#' 
#' Usage: Rscript generate_data.R [start_year] [end_year]

cat("=== First and Flow: NFL Data Generation Pipeline ===\n")
cat("Generating data for flow matching model training...\n\n")

# Source the main data processor
source("nfl_data_processor.R")

# Default parameters optimized for flow matching
DEFAULT_START_YEAR <- 2018  # More recent data for better model performance
DEFAULT_END_YEAR <- 2023
DEFAULT_OUTPUT_FILE <- "processed/nfl_flow_training_data.csv"

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
start_year <- if(length(args) >= 1) as.numeric(args[1]) else DEFAULT_START_YEAR
end_year <- if(length(args) >= 2) as.numeric(args[2]) else DEFAULT_END_YEAR

# Create processed data directory
if (!dir.exists("processed")) {
  dir.create("processed", recursive = TRUE)
  cat("Created processed data directory.\n")
}

# Create raw data directory for caching
if (!dir.exists("raw")) {
  dir.create("raw", recursive = TRUE)
  cat("Created raw data directory.\n")
}

cat("Generating training data for flow matching model...\n")
cat("Years:", start_year, "to", end_year, "\n")
cat("Output:", DEFAULT_OUTPUT_FILE, "\n\n")

# Execute the data processing pipeline
system(paste("Rscript nfl_data_processor.R", start_year, end_year, DEFAULT_OUTPUT_FILE))

# Verify output
if (file.exists(DEFAULT_OUTPUT_FILE)) {
  cat("\n✅ Data generation completed successfully!\n")
  cat("Training data saved to:", DEFAULT_OUTPUT_FILE, "\n")
  
  # Quick data summary
  if (require("readr", quietly = TRUE)) {
    data <- read_csv(DEFAULT_OUTPUT_FILE, show_col_types = FALSE)
    cat("Dataset summary:\n")
    cat("  Rows:", nrow(data), "\n")
    cat("  Columns:", ncol(data), "\n")
    cat("  Seasons:", min(data$season), "to", max(data$season), "\n")
    cat("  Teams:", length(unique(data$team)), "\n")
  }
} else {
  cat("\n❌ Data generation failed. Check error messages above.\n")
}

cat("\nData ready for flow matching model development in src/\n")
