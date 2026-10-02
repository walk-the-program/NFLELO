# NFL ELO Rating System Analysis Tool

A comprehensive Python script for analyzing NFL team performance using ELO ratings. This tool processes historical game data, calculates various performance metrics, and generates visualizations to understand team performance, league parity, luck factors, and forecast calibration.

## Overview

The `NFL_ELO_organized.py` script provides:
- **ELO Rating System**: Calculates team ELO ratings with home field advantage, margin of victory scaling, and season reset mechanisms
- **Comprehensive Analytics**: Luck index, calibration curves, upset analysis, parity metrics, strength of schedule
- **Rich Visualizations**: 20+ different plots with custom NFL-inspired color schemes
- **Configurable**: Toggle plots on/off, customize parameters, control display
- **Historical Analysis**: Covers NFL history from 1970 to present

## Features

### Core Functionality
- ELO rating calculations with configurable parameters
- Margin of victory (MOV) scaling with multiple methods
- Season reset/regression to prevent ELO drift
- League-wide mean tracking
- Team timeline and monthly aggregation

### Advanced Metrics
- **Luck Index**: Measures how much teams over/underperform relative to ELO expectations
- **Parity Metrics**: Analyzes league competitiveness over time
- **Calibration Analysis**: Evaluates forecast accuracy using Brier scores and log loss
- **Strength of Schedule**: Calculates opponent difficulty using actual ELO ratings
- **Upset Analysis**: Tracks and visualizes unexpected game outcomes

### Visualizations
- League-wide tapestry showing all teams over time
- Individual team trajectory analysis (small multiples)
- Season ladder rankings
- Team summary statistics
- Biggest ELO swings analysis
- Parity metrics over time
- Luck index visualizations
- Calibration curves and Brier score decomposition
- Strength of schedule vs ELO scatter plots
- And many more...

## Requirements

### Python Packages
```
pandas
numpy
matplotlib
seaborn
openpyxl  # For reading Excel files
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn openpyxl
```

### Data File
- **Input File**: `NFLELO_data.xlsx` must be in the same directory as the script
- **Required Columns**: Date, Team, Opp, Result, Hosting
- **Date Format**: MM/DD/YY
- **Result Format**: "W 24-17", "L 14-21", "T 10-10"
- **Hosting Format**: "H" (home), "@" (away), "N" (neutral)

## Usage

### Basic Usage
```bash
python NFL_ELO_organized.py
```

The script will:
1. Load data from `NFLELO_data.xlsx`
2. Calculate ELO ratings for all teams
3. Generate analysis metrics
4. Create visualizations (if enabled)
5. Save output files to the `Outputs/` directory

### Configuration

The script uses a `NFLConfig` class for all configuration. Modify settings at the top of the script:

```python
config = NFLConfig()

# Example: Adjust ELO parameters
config.K_FACTOR = 20.0
config.HOME_FIELD_ADVANTAGE = 55.0
config.SEASON_RESET_LAMBDA = 0.15

# Example: Toggle specific plots
config.toggle_plot('league_tapestry', enabled=True)
config.toggle_plot('team_small_multiples', enabled=False)

# Example: Disable all plots for quick analysis
config.disable_all_plots()
```

### Key Configuration Options

#### ELO Parameters
- `START_ELO`: Initial ELO rating (default: 1000.0)
- `K_FACTOR`: Rating volatility factor (default: 20.0)
- `HOME_FIELD_ADVANTAGE`: Home field advantage in ELO points (default: 55.0)
- `USE_MARGIN_OF_VICTORY`: Enable MOV scaling (default: True)
- `MOV_SCALING_METHOD`: "elo_standard", "log", or "sqrt" (default: "elo_standard")
- `USE_SEASON_RESET`: Enable season reset/regression (default: True)
- `SEASON_RESET_LAMBDA`: Regression factor (0.0 = no reset, 1.0 = full reset, default: 0.15)

#### Plot Toggles
All plots can be enabled/disabled individually:
- `league_tapestry`: League-wide team trajectories
- `team_small_multiples`: Individual team charts
- `season_ladder`: Season rankings
- `luck_index`: Luck analysis visualizations
- `calibration_plots`: Forecast calibration curves
- And many more...

## Output Files

All outputs are saved to the `Outputs/` directory:

### CSV Files
- `comprehensive_nfl_data.csv`: Complete dataset with all metrics
- `luck_index_full.csv`: Full luck index data by team and season
- `average_luck_by_team.csv`: Aggregated luck scores
- `sensitivity_analysis_results.csv`: Parameter sensitivity results

### PNG Files
- `league_tapestry.png`: All teams over time
- `team_small_multiples.png`: Individual team charts
- `season_ladder_2024.png`: Current season rankings
- `luck_index_2024.png`: Luck analysis
- `calibration_plots.png`: Forecast accuracy
- And many more visualization files...

## Code Structure

The script is organized into logical sections:

1. **Imports and Setup** (lines 37-62)
   - Library imports and configuration

2. **Configuration System** (lines 64-378)
   - `NFLConfig` class with all settings

3. **Utility Functions** (lines 382-551)
   - `safefig`: Safe figure handling decorator
   - `season_from_date`: Convert dates to NFL seasons
   - `canonical_franchise`: Map historical teams to current franchises
   - `add_rolling_features`: Add rolling statistics

4. **Data Loading** (lines 553-620)
   - `load_nfl_data`: Load and preprocess Excel data

5. **ELO Calculations** (lines 622-813)
   - `calculate_elo_ratings`: Main ELO calculation function

6. **Data Processing** (lines 815-1020)
   - `create_team_timeline`: Create team ELO timeline
   - `create_monthly_elo_data`: Monthly aggregation
   - `create_elo_games_data`: Game-level data structure

7. **Analysis Functions** (lines 1022-2650)
   - Team summary statistics
   - Biggest swings analysis
   - Season summaries
   - Luck index calculations
   - Calibration analysis
   - Parity metrics
   - Strength of schedule
   - Sensitivity analysis

8. **Plotting Functions** (lines 2650-4354)
   - All visualization functions

9. **Main Execution** (lines 4356-4517)
   - `main()`: Orchestrates the entire analysis pipeline

## Key Functions

### Data Loading
```python
raw_data = load_nfl_data()  # Loads from NFLELO_data.xlsx
```

### ELO Calculation
```python
elo_games = calculate_elo_ratings(raw_data)
timeline = create_team_timeline(elo_games)
```

### Analysis
```python
summary_stats = calculate_team_summary_stats(timeline)
luck_index = build_luck_index(timeline, elo_games)
parity_metrics = build_parity_metrics_clean(season_team_summary, elo_games)
```

### Visualization
```python
plot_league_tapestry(team_monthly_elo)
plot_luck_index(luck_index)
plot_calibration_and_brier(calibration_curve, calibration_perf)
```

## Mathematical Formulas

### ELO Expected Win Probability
```
Ea = 1 / (1 + 10^((Rb_eff - Ra_eff) / 400))
```
Where `Ra_eff` and `Rb_eff` include home field advantage.

### ELO Update
```
Delta = K * MOV_Factor * (Actual - Expected)
New_Rating = Old_Rating + Delta
```

### Margin of Victory Factor
```
MOV_Factor = log(margin + 1) * (2.2 / ((|Ra_eff - Rb_eff| * 0.001 + 2.2)))
```

### Season Reset
```
New_Rating = (1 - λ) * End_Season_Rating + λ * START_ELO
```

## Customization Examples

### Change ELO Parameters
```python
config.K_FACTOR = 25.0  # More volatile ratings
config.HOME_FIELD_ADVANTAGE = 60.0  # Stronger home advantage
config.SEASON_RESET_LAMBDA = 0.20  # More regression to mean
```

### Disable Specific Plots
```python
config.toggle_plot('team_small_multiples', enabled=False)
config.toggle_plot('luck_timelines', enabled=False)
```

### Run Analysis Without Visualizations
```python
config.disable_all_plots()
# Then run the script
```

## Troubleshooting

### File Not Found Error
- Ensure `NFLELO_data.xlsx` is in the same directory as the script
- Check that the file name matches exactly (case-sensitive)

### Import Errors
- Install required packages: `pip install pandas numpy matplotlib seaborn openpyxl`
- Ensure you're using Python 3.7 or higher

### Memory Issues
- Disable plots you don't need: `config.disable_all_plots()`
- Process data in smaller chunks if working with very large datasets

## Performance Tips

1. **Disable Unused Plots**: Use `config.disable_all_plots()` for faster analysis
2. **Sensitivity Analysis**: Can take 5-10 minutes for full grid search (240 combinations)
3. **Large Datasets**: The script is optimized but may take time with very large datasets

## Author

Walker Tracy  
Created: October 17th, 2025

## License

This script is provided as-is for analysis purposes.

## Notes

- The script handles team relocations and name changes throughout NFL history
- Historical conference/division assignments are accounted for (e.g., 2002 realignment)
- All dates are converted to NFL seasons (Jan/Feb games belong to previous season)
- The script automatically creates the `Outputs/` directory if it doesn't exist

