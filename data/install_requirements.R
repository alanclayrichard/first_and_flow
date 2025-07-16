# R Package Requirements for NFL Data Processing
# 
# Required packages for nfl_data_processor.R
# Install these manually if the auto-installation fails:

# Primary packages
install.packages("nflfastR")   # NFL play-by-play data
install.packages("dplyr")      # Data manipulation  
install.packages("readr")      # CSV reading/writing
install.packages("stringr")    # String manipulation

# Optional packages for enhanced functionality
install.packages("data.table") # Alternative fast data manipulation
install.packages("lubridate")  # Date handling
install.packages("ggplot2")    # Data visualization

# Development packages (optional)
install.packages("testthat")   # Unit testing
install.packages("devtools")   # Development tools

# =============================================================================
# SYSTEM REQUIREMENTS
# =============================================================================

# R version: >= 4.0.0 recommended
# Memory: >= 4GB RAM for processing multiple seasons
# Storage: ~100MB per season of data
# Internet: Required for downloading nflfastR data

# =============================================================================
# TROUBLESHOOTING
# =============================================================================

# If nflfastR installation fails, try:
# install.packages("nflfastR", dependencies = TRUE, type = "binary")

# For macOS users with M1/M2 chips:
# May need to install Xcode command line tools:
# xcode-select --install

# For dependency issues:
# install.packages(c("curl", "openssl", "httr", "jsonlite"))
