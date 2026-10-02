# NFL ELO Analysis Tool - Complete Documentation

## Table of Contents
1. [Overview](#overview)
2. [Code Structure](#code-structure)
3. [Configuration System](#configuration-system)
4. [Data Structures](#data-structures)
5. [Core Calculations](#core-calculations)
6. [Plotting Functions & Outputs](#plotting-functions--outputs)
7. [Output Interpretations](#output-interpretations)
8. [Usage Guide](#usage-guide)

---

## Overview

### Purpose
The NFL ELO Analysis Tool is a comprehensive system for analyzing NFL team performance using ELO ratings. It processes historical game data, calculates various performance metrics, and generates visualizations to understand team performance, league parity, luck factors, and forecast calibration.

### Key Features
- **ELO Rating System**: Calculates team ELO ratings with home field advantage
- **Comprehensive Analytics**: Luck index, calibration curves, upset analysis, parity metrics
- **Rich Visualizations**: 15+ different plots with custom color schemes
- **Configurable**: Toggle plots on/off, customize parameters, control display
- **Historical Analysis**: Covers NFL history from 1970 to present

### Data Requirements
- **Input File**: `NFLELO_data.xlsx` in the same directory
- **Required Columns**: Date, Team, Opp, Result, Hosting
- **Date Format**: MM/DD/YY
- **Result Format**: "W 24-17", "L 14-21", "T 10-10"
- **Hosting Format**: "H" (home), "@" (away), "N" (neutral)

---

## Code Structure

### File Organization
```
NFL_ELO_organized.py
├── Imports and Setup (lines 37-62)
├── Configuration System (lines 64-344)
│   └── NFLConfig class
├── Utility Functions (lines 346-469)
│   ├── safefig decorator
│   ├── season_from_date
│   ├── canonical_franchise
│   └── add_rolling_features
├── Data Loading & Preprocessing (lines 471-794)
│   ├── load_nfl_data
│   ├── calculate_elo_ratings
│   ├── create_team_timeline
│   ├── create_monthly_elo_data
│   └── create_elo_games_data
├── Analysis Functions (lines 796-1296)
│   ├── calculate_team_summary_stats
│   ├── find_biggest_swings
│   ├── create_season_team_summary
│   ├── create_season_conference_summary
│   ├── build_luck_index
│   ├── build_calibration_bins
│   ├── build_upset_map
│   ├── build_sos
│   └── build_parity_metrics_clean
├── Plotting Functions (lines 1298-2549)
│   ├── plot_league_tapestry
│   ├── plot_team_small_multiples
│   ├── plot_season_ladder
│   ├── plot_team_summary_stats
│   ├── plot_biggest_swings
│   ├── plot_parity_metrics
│   ├── plot_combined_parity
│   ├── plot_season_delta_normalized
│   ├── plot_conference_balance
│   ├── plot_luck_index
│   ├── plot_calibration_and_brier
│   ├── plot_any_given_sunday_line
│   ├── plot_luck_timelines
│   ├── plot_sos_vs_elo
│   └── export_team_story_card
└── Main Execution (lines 2551-2671)
    └── main function
```

### Key Design Patterns

#### 1. **Configuration-Driven Architecture**
- All settings centralized in `NFLConfig` class
- Plot toggles for performance optimization
- Plot-specific configurations for customization

#### 2. **Decorator Pattern**
- `@safefig` decorator ensures proper figure cleanup
- Prevents memory leaks from unclosed matplotlib figures

#### 3. **Data Pipeline**
```
Raw Data → ELO Games → Timeline → Monthly Aggregation → Analysis → Visualization
```

#### 4. **Modular Functions**
- Each analysis function is independent
- Functions can be called individually or in sequence
- Easy to extend with new analyses

---

## Configuration System

### NFLConfig Class

The `NFLConfig` class centralizes all configuration settings:

#### Core ELO Parameters
```python
START_ELO = 1000.0              # Initial ELO rating for all teams
K_FACTOR = 20.0                  # ELO update sensitivity (higher = more volatile)
HOME_FIELD_ADVANTAGE = 55.0      # ELO points added for home team
ROLLING_WINDOW_MONTHS = 3        # Window for rolling averages
```

### Season Reset Parameters
```python
SEASON_RESET_LAMBDA = 0.15       # Regression to mean factor (0.0 = no reset, 1.0 = full reset)
USE_SEASON_RESET = True          # Enable season reset/regression
```

### Margin of Victory Parameters
```python
USE_MARGIN_OF_VICTORY = True     # Enable MOV scaling
MOV_SCALING_METHOD = "elo_standard"  # "elo_standard", "log", "sqrt"
```

### League Mean Anchoring
```python
TRACK_LEAGUE_MEAN = True         # Track league-wide mean ELO
RECENTER_ELO = False             # Re-center ELO to START_ELO after each game (not recommended)
```

### Parity Index Configuration
```python
PARITY_WEIGHTS = {
    "dispersion": 0.35,
    "rank_stab": 0.25,
    "upset_gap": 0.25,
    "entropy": 0.15,
}
PARITY_ANCHOR_ERA_START = 1970   # Start era for parity scaling
PARITY_ANCHOR_ERA_END = 1979     # End era for parity scaling
PARITY_SCALE_MIN = 0             # Minimum parity score
PARITY_SCALE_MAX = 100           # Maximum parity score
```

### Pythagorean Expectation
```python
PYTHAG_EXPONENT = 2.37           # NFL exponent for Pythagorean expectation
```

### Close Game Definition
```python
CLOSE_GAME_POINT_DIFF = 7        # Point differential threshold
CLOSE_GAME_PROB_RANGE = (0.35, 0.65)  # Win probability range for close games
```

#### Color Scheme
The system uses a custom color palette:
- **Primary Colors**: Deep navy blue, charcoal gray, NFL red
- **Series Colors**: 12-color palette for multiple series
- **Team Colors**: Official NFL team colors (32 teams)
- **Background/Text**: White background, dark text, light grid

#### Plot Toggles
All 24 plots can be individually enabled/disabled:
- `league_tapestry`, `team_small_multiples`, `season_ladder`
- `team_summary`, `biggest_swings`, `parity_metrics`
- `combined_parity`, `delta_normalized`, `conference_balance`
- `luck_index`, `calibration_plots`, `upset_heatmap`
- `luck_timelines`, `sos_vs_elo`, `story_card`
- And more...

#### Display Settings
```python
SHOW_PLOTS = False  # Set to False to disable plot display (plots still saved)
OUTPUT_DIR = Path("Outputs")  # Directory for saved plots
```

#### Configuration Methods
```python
config.is_plot_enabled('plot_name')  # Check if plot is enabled
config.get_plot_config('plot_name')  # Get plot-specific config
config.toggle_plot('plot_name')      # Toggle plot on/off
config.disable_all_plots()            # Disable all plots
config.enable_all_plots()             # Enable all plots
config.get_team_color('KAN')          # Get team color
config.apply_color_scheme(ax)         # Apply color scheme to axes
```

---

## Data Structures

### 1. Raw Data (`df`)
**Source**: `NFLELO_data.xlsx`

**Columns**:
- `Date`: Game date (datetime)
- `Team`: Team code (e.g., "KAN", "NWE")
- `Opp`: Opponent code
- `Result`: Game result ("W 24-17", "L 14-21", "T 10-10")
- `Hosting`: Home/Away/Neutral ("H", "@", "N")
- `TeamPoints`: Points scored by team
- `OppPoints`: Points scored by opponent
- `PointDiff`: Point differential
- `Team_FR`: Canonical franchise code (handles relocations)
- `Opp_FR`: Opponent canonical franchise code
- `Season`: NFL season (Jan/Feb games belong to previous season)

### 2. ELO Games (`elo_games`)
**Source**: `calculate_elo_ratings()`

**Structure**: One row per game, with both teams' data

**Columns**:
- `Date`, `Season`, `GameID`
- `TeamA`, `TeamB`: Team codes
- `A_Pre`, `B_Pre`: Pre-game ELO ratings
- `A_Exp`, `B_Exp`: Expected win probabilities (0-1)
- `A_Act`, `B_Act`: Actual outcomes (1=win, 0.5=tie, 0=loss)
- `A_Res`, `B_Res`: Same as A_Act/B_Act (for compatibility)
- `A_Delta`, `B_Delta`: ELO change from game
- `A_Post`, `B_Post`: Post-game ELO ratings
- `A_Home`, `B_Home`: Home field indicator (1=home, 0=away)
- `A_Points`, `B_Points`: Points scored
- `PointDiff`: Point differential (A - B)
- `MOV_Factor`: Margin of victory scaling factor (if enabled)
- `LeagueMean`: League-wide mean ELO (if tracking enabled)

**Purpose**: Foundation for all ELO calculations and game-level analysis

### 3. Timeline (`timeline`)
**Source**: `create_team_timeline()`

**Structure**: One row per team per game (2 rows per game)

**Columns**:
- `Date`, `Season`, `Team`, `Opponent`
- `Elo_Pre`: ELO rating before game
- `Elo_Post`: ELO rating after game
- `Elo_Change`: Change in ELO (delta)
- `Expected`: Expected win probability
- `Actual`: Actual outcome (1=win, 0.5=tie, 0=loss)
- `Home`: Home field indicator
- `Points_For`: Points scored by team
- `Points_Against`: Points allowed
- `Outcome`: Game outcome ("W", "L", "T")
- Rolling features: `Elo_Post_RollMean`, `Elo_Post_RollMed`, `Elo_Post_RollStd`

**Purpose**: Team-centric view of all games, enables team trajectory analysis

### 4. Monthly ELO Data (`team_monthly_elo`)
**Source**: `create_monthly_elo_data()`

**Structure**: One row per team per month

**Columns**:
- `Franchise`: Team code (renamed from Team)
- `YearMonth`: Monthly timestamp
- `Elo_Last`: Last ELO rating in month
- `Elo_Mean`: Average ELO rating in month
- `Date_Last`: Date of last game in month
- Rolling features: `Elo_Last_RollMean`, `Elo_Mean_RollMean`, etc.

**Purpose**: Smoothed data for visualization (tapestry plots)

### 5. Season Team Summary (`season_team_summary`)
**Source**: `create_season_team_summary()` + `calculate_volatility_and_dynasty()`

**Structure**: One row per team per season

**Columns**:
- `Season`, `Team`
- `Elo_Pre`: Start-of-season ELO
- `Elo_Post`: End-of-season ELO
- `Games`: Games played
- `Wins`, `Losses`, `Ties`
- `Start_Rank`: Ranking at season start
- `End_Rank`: Ranking at season end
- `Volatility`: Standard deviation of game-to-game ELO changes
- `DynastyScore`: Sum of ELO above 1500 threshold (normalized)
- `Max_ELO`, `Min_ELO`, `ELO_Range`: ELO statistics for season

**Purpose**: Season-level aggregations for analysis

### 6. Season Conference Summary (`season_conference_summary`)
**Source**: `create_season_conference_summary()`

**Structure**: One row per conference per season

**Columns**:
- `Season`, `Conference` ("AFC" or "NFC")
- `EloMean`: Average conference ELO
- `EloEnd`: End-of-season average ELO
- `TeamCount`: Number of teams

**Purpose**: Conference-level analysis

### 7. Luck Index (`luck_df`)
**Source**: `build_luck_index()`

**Structure**: One row per team per season

**Columns**:
- `Season`, `Team`
- `Games`: Games played
- `ActualWins`: Actual wins
- `ExpWins`: Expected wins (sum of win probabilities)
- `Luck_Index`: ActualWins - ExpWins
- `Luck_per_Game`: Luck_Index / Games
- `Close_ActualWins`: Wins in close games
- `Close_ExpWins`: Expected wins in close games
- `Close_Games`: Number of close games
- `Close_Luck_Index`: Close_ActualWins - Close_ExpWins
- `Close_Luck_per_Game`: Close_Luck_Index / Close_Games
- `Points_For`, `Points_Against`: Total points
- `Pyth_ExpWins`: Pythagorean expected wins
- `Pyth_Luck`: ActualWins - Pyth_ExpWins

**Purpose**: Measure of over/under-performance vs. expectations (ELO and Pythagorean)

### 8. Calibration Data (`calibration_curve`, `calibration_perf`)
**Source**: `build_calibration_bins()`

**Calibration Curve**:
- `Bin`: Quantile bin of predicted win probability
- `Pwin_mean`: Mean predicted win probability in bin
- `Outcome_mean`: Mean actual outcome in bin
- `N`: Number of games in bin

**Season Performance**:
- `Season`
- `Brier`: Brier score (mean squared error)
- `LogLoss`: Logarithmic loss

**Purpose**: Evaluate forecast calibration and accuracy

### 9. Upset Map (`upset_map`)
**Source**: `build_upset_map()`

**Structure**: One row per season per ELO difference bin

**Columns**:
- `Season`, `Bin`: ELO difference bin (e.g., 0-25, 25-50)
- `UpsetRate`: Actual upset rate (underdog wins)
- `ExpUpset`: Expected upset rate
- `Gap`: UpsetRate - ExpUpset (positive = more chaos)
- `N`: Number of games in bin

**Purpose**: Analyze "Any Given Sunday" effect by ELO difference

### 10. Strength of Schedule (`sos_data`)
**Source**: `build_sos()`

**Structure**: One row per team per season

**Columns**:
- `Season`, `Team`
- `SoS_PreMean`: Mean opponent ELO before game
- `SoS_EndMean`: Mean opponent end-of-season ELO
- `SoS_PreWeighted`: Mean opponent ELO (weighted by home/away)
- `SoS_EndWeighted`: Mean opponent end-of-season ELO (weighted by home/away)
- `Games`: Games played
- `Elo_Post`: End-of-season ELO (merged from season_team_summary)

**Purpose**: Measure schedule difficulty using multiple metrics

### 11. Parity Metrics (`parity_metrics`)
**Source**: `build_parity_metrics_clean()`

**Structure**: One row per season

**Columns**:
- `Season`, `Teams`
- `Dispersion_STD`: Standard deviation of end-of-season ELO
- `RankStability_Spearman`: Correlation between start/end rankings
- `Upset_Gap`: Actual upsets - Expected upsets
- `Forecast_Entropy`: Average entropy of win probabilities
- `ELO_Mean`, `ELO_Range`: Mean and range of ELO ratings
- `Top_Team`, `Bottom_Team`: Best and worst teams
- `Z_Dispersion`, `Z_RankStab`, `Z_UpsetGap`, `Z_Entropy`: Z-scores
- `Parity_Index`: Weighted composite parity score (z-score based)
- `Parity_Score`: Rescaled parity index (0-100 scale, anchored to reference era)

**Purpose**: Comprehensive league parity analysis

### 12. Forecast Sharpness (`sharpness_metrics`)
**Source**: `calculate_forecast_sharpness()`

**Structure**: One row per season

**Columns**:
- `Season`
- `Sharpness`: Standard deviation of predicted win probabilities
- `Mean_Entropy`: Average entropy of win probabilities
- `Games`: Number of games

**Purpose**: Measure forecast confidence (high sharpness = strong calls, low = timid)

### 13. Division Summary (`division_summary`)
**Source**: `create_division_summary()`

**Structure**: One row per division per season

**Columns**:
- `Season`, `Division`, `Conference`
- `Teams`: Number of teams in division
- `Mean_ELO`: Average division ELO
- `Std_ELO`: Standard deviation of division ELO
- `Range_ELO`: ELO range within division
- `Parity_Score`: Division parity score (lower std = higher parity)
- `Top_Team`, `Bottom_Team`: Best and worst teams in division
- `Division_Rank`: Division strength ranking by season

**Purpose**: Division-level analysis and parity metrics

---

## Core Calculations

### 1. ELO Rating Calculation

#### Formula
For each game between Team A and Team B:

1. **Get Pre-Game Ratings**:
   ```
   Ra = ELO rating of Team A
   Rb = ELO rating of Team B
   ```

2. **Apply Home Field Advantage**:
   ```
   HFA = 55.0 (configurable)
   Ra_eff = Ra + HFA (if Team A is home)
   Rb_eff = Rb + HFA (if Team B is home)
   ```

3. **Calculate Expected Win Probabilities**:
   ```
   Ea = 1 / (1 + 10^((Rb_eff - Ra_eff) / 400))
   Eb = 1 - Ea
   ```
   - Ea = probability Team A wins
   - Eb = probability Team B wins
   - Formula based on logistic curve with 400-point scale

4. **Determine Actual Outcomes**:
   ```
   Sa = 1.0 if Team A wins
   Sa = 0.5 if tie
   Sa = 0.0 if Team A loses
   Sb = 1 - Sa (unless tie, then Sb = 0.5)
   ```

5. **Calculate Margin of Victory Factor** (if enabled):
   ```
   margin = abs(PointDiff)
   MOV_Factor = log(margin + 1) * (2.2 / ((abs(Ra_eff - Rb_eff) * 0.001 + 2.2)))
   MOV_Factor = min(MOV_Factor, 2.0)  # Cap at 2.0
   ```
   Alternative methods: `log` or `sqrt` scaling (configurable)

6. **Calculate ELO Changes**:
   ```
   Delta_A = K * MOV_Factor * (Sa - Ea)
   Delta_B = K * MOV_Factor * (Sb - Eb)
   ```

7. **Update Ratings**:
   ```
   Ra_new = Ra + Delta_A
   Rb_new = Rb + Delta_B
   ```

8. **Season Reset** (at season boundary, if enabled):
   ```
   λ = SEASON_RESET_LAMBDA (default: 0.15)
   Elo_new_season_start = (1 - λ) * Elo_end_last_season + λ * START_ELO
   ```

#### Interpretation
- **Higher ELO = Better Team**: Ratings typically range from ~800 (worst) to ~1800 (best)
- **400-Point Rule**: 400-point difference ≈ 90% win probability
- **K-Factor**: Controls rating volatility (higher = more responsive to results)
- **Home Field**: Adds 55 points to home team's effective rating
- **Margin of Victory**: Large blowouts result in more rating movement
- **Season Reset**: Prevents runaway drift, anchors to current era (λ=0.15 means 15% regression to mean)

### 2. Luck Index Calculation

#### Formula
For each team in each season:

```
ActualWins = sum of actual wins (1.0 for win, 0.5 for tie)
ExpWins = sum of Expected win probabilities (from ELO model)
Luck_Index = ActualWins - ExpWins
Luck_per_Game = Luck_Index / Games
```

#### Interpretation
- **Positive Luck_Index**: Team won more games than expected (good luck, clutch performance, or model underestimation)
- **Negative Luck_Index**: Team won fewer games than expected (bad luck, poor execution in close games, or model overestimation)
- **Zero**: Team performed exactly as expected
- **Typical Range**: -3 to +3 wins per season

#### Close-Game Luck
Close games defined as:
- Point differential ≤ 7, OR
- Predicted win probability in [0.35, 0.65]

```
Close_Luck_Index = Close_ActualWins - Close_ExpWins
```

#### Pythagorean Luck
```
PF = total points for
PA = total points against
γ = 2.37 (NFL exponent, configurable)

Pyth_WinPct = PF^γ / (PF^γ + PA^γ)
Pyth_ExpWins = Pyth_WinPct * Games
Pyth_Luck = ActualWins - Pyth_ExpWins
```

#### Use Cases
- Identify teams that over/under-performed expectations
- Analyze "clutch factor" vs. "luck" (close-game luck)
- Compare ELO luck vs. Pythagorean luck (model mis-rating vs. true luck)
- Evaluate model calibration by team

### 3. Calibration Analysis

#### Reliability Curve
Bins games by predicted win probability (quantiles), then compares:
- **X-axis**: Mean predicted win probability in bin
- **Y-axis**: Mean actual outcome in bin
- **Perfect Calibration**: Points lie on y=x line
- **Overconfident**: Points below line (predicted too high)
- **Underconfident**: Points above line (predicted too low)

#### Segmented Calibration
Calibration can be segmented by:
- **Era**: 1970s, 1980s, etc. (identify era-specific issues)
- **ELO Difference**: 0-50, 50-100, etc. (check if model overconfident for big favorites)
- **Home/Away**: Separate calibration for home vs. away games

Usage:
```python
build_calibration_bins(elo_games, groupby_cols=["Era10"])
build_calibration_bins(elo_games, groupby_cols=["EloDiffBin"])
build_calibration_bins(elo_games, groupby_cols=["HomeFlag"])
```

#### Brier Score
```
Brier = mean((Actual - Predicted)^2)
```
- **Range**: 0 (perfect) to 1 (worst)
- **Lower is Better**: Measures forecast accuracy
- **Typical Values**: 0.15-0.25 for NFL

#### Log Loss
```
LogLoss = -mean(Actual * log(Predicted) + (1-Actual) * log(1-Predicted))
```
- **Range**: 0 (perfect) to infinity
- **Lower is Better**: Penalizes confident wrong predictions more
- **Typical Values**: 0.3-0.7 for NFL

### 4. Upset Analysis

#### Definitions
- **Underdog**: Team with lower pre-game ELO rating
- **Upset**: Underdog wins the game (ties excluded)
- **Expected Upset Rate**: Mean of min(A_Exp, B_Exp) for each game

#### Calculation
For each ELO difference bin (e.g., 0-25, 25-50, 50-75 points):

```
UpsetRate = mean(upset indicator)  # Actual upset rate
ExpUpset = mean(min(A_Exp, B_Exp))  # Expected upset rate
Gap = UpsetRate - ExpUpset          # Difference
```

#### Interpretation
- **Positive Gap**: More upsets than expected ("Any Given Sunday" effect)
- **Negative Gap**: Fewer upsets than expected (favorites win more)
- **Larger ELO Differences**: Should have fewer upsets (larger favorites)
- **Small ELO Differences**: Should have more upsets (close games)

### 5. Parity Metrics

#### Four Components

**1. Dispersion (Standard Deviation)**
```
Dispersion_STD = std(End-of-Season ELO ratings)
Z_Dispersion = -z_score(Dispersion_STD)  # Negative because lower std = more parity
```
- **Lower STD = More Parity**: Teams are closer in skill
- **Higher STD = Less Parity**: Clear hierarchy

**2. Rank Stability (Spearman Correlation)**
```
RankStability = corr(Start_Rank, End_Rank, method='spearman')
Z_RankStab = -z_score(RankStability)  # Negative because lower correlation = more parity
```
- **Lower Correlation = More Parity**: Rankings change more (competitive balance)
- **Higher Correlation = Less Parity**: Rankings stable (predictable hierarchy)

**3. Upset Gap**
```
Upset_Gap = Actual_Upsets - Expected_Upsets
Z_UpsetGap = z_score(Upset_Gap)  # Positive because more upsets = more parity
```
- **Positive = More Parity**: More underdogs win (competitive)
- **Negative = Less Parity**: Favorites win more (predictable)

**4. Forecast Entropy**
```
Entropy = -p*log(p) - (1-p)*log(1-p)  # For each game's win probability
Forecast_Entropy = mean(Entropy)
Z_Entropy = z_score(Forecast_Entropy)  # Positive because higher entropy = more parity
```
- **Higher Entropy = More Parity**: Win probabilities closer to 50/50 (uncertain)
- **Lower Entropy = Less Parity**: Win probabilities extreme (predictable)

#### Composite Parity Index
```
Parity_Index = weighted_average(Z_Dispersion, Z_RankStab, Z_UpsetGap, Z_Entropy)
```
- **Weights**: Configurable via `config.PARITY_WEIGHTS` (default: 0.35, 0.25, 0.25, 0.15)
- **Higher = More Parity**: More competitive, unpredictable league
- **Lower = Less Parity**: More predictable, hierarchical league

#### Parity Score (0-100 Scale)
```
Parity_Score = 50 + (Parity_Index - anchor_mean) / anchor_std * 10
```
- **Anchored to Reference Era**: 1970s average = 50 (configurable)
- **Range**: 0-100 (clipped)
- **More Intuitive**: Easier to interpret than z-scores
- **50 = Reference Era Parity**: Values above/below indicate more/less parity

### 6. Strength of Schedule (SoS)

#### Implementation
Four SoS metrics are calculated:

1. **SoS_PreMean**: Mean opponent ELO before game
   ```
   SoS_PreMean = mean(Opp_Elo_Pre for all games)
   ```

2. **SoS_EndMean**: Mean opponent end-of-season ELO
   ```
   SoS_EndMean = mean(Opp_Elo_EndSeason for all games)
   ```

3. **SoS_PreWeighted**: Mean opponent ELO (weighted by home/away)
   ```
   weight = 1.15 if away, 1.0 if home
   SoS_PreWeighted = mean(Opp_Elo_Pre * weight)
   ```

4. **SoS_EndWeighted**: Mean opponent end-of-season ELO (weighted)
   ```
   SoS_EndWeighted = mean(Opp_Elo_EndSeason * weight)
   ```

#### Interpretation
- **Higher SoS = Harder Schedule**: Faced stronger opponents
- **Lower SoS = Easier Schedule**: Faced weaker opponents
- **PreMean vs EndMean**: PreMean uses opponent strength at time of game; EndMean uses final opponent strength
- **Weighted versions**: Account for difficulty of away games

### 7. Season Delta Normalized

#### Calculation
For each team in a season:
```
Delta = Elo_Post - Elo_Pre  # Change over season
Normalized_Delta = (Delta - mean(Delta)) / std(Delta)  # Z-score
```

#### Interpretation
- **Positive**: Team improved more than average
- **Negative**: Team declined more than average
- **Zero**: Average change
- **Normalized**: Allows comparison across seasons

### 8. Conference Balance

#### Calculation
For each season:
```
AFC_Mean = mean(End-of-Season ELO for AFC teams)
NFC_Mean = mean(End-of-Season ELO for NFC teams)
Balance = AFC_Mean - NFC_Mean  # or absolute difference
```

#### Interpretation
- **Positive**: AFC stronger
- **Negative**: NFC stronger
- **Zero**: Perfect balance
- **Trends**: Shows which conference dominates over time

### 9. ELO Volatility and Dynasty Score

#### Volatility
```
Volatility = std(Elo_Change) for all games in season
```
- **High Volatility**: Boom & bust teams, wild swings
- **Low Volatility**: Steady grinders, consistent performance

#### Dynasty Score
```
DynastyScore = sum(max(0, Elo_Post - 1500)) / 100
```
- **Measures Sustained Dominance**: Sum of ELO above threshold
- **Normalized by 100**: More interpretable scale
- **Higher = More Dominant**: Teams that maintain high ELO over time

### 10. Forecast Sharpness

#### Calculation
```
Sharpness = std(A_Exp) across all games in season
Mean_Entropy = mean(-p*log(p) - (1-p)*log(1-p))
```

#### Interpretation
- **High Sharpness**: Model makes strong calls (probabilities spread out)
- **Low Sharpness**: Model is timid (probabilities cluster around 50%)
- **Combined with Calibration**: Evaluates if forecasts are both sharp AND well-calibrated

### 11. Division-Level Analysis

#### Division Metrics
For each division per season:
```
Mean_ELO = mean(Elo_Post for all teams in division)
Std_ELO = std(Elo_Post for all teams in division)
Parity_Score = 1.0 / (1.0 + Std_ELO / 100.0)
Division_Rank = rank(Mean_ELO) among all divisions
```

#### Interpretation
- **Division Strength**: Mean ELO shows overall division quality
- **Division Parity**: Lower std = more balanced division
- **Rankings**: Compare division strength across seasons

---

## Plotting Functions & Outputs

### 1. League Tapestry (`plot_league_tapestry`)
**Output**: `Outputs/league_tapestry.png`

**Visualization**: Line graph showing all 32 teams' ELO ratings over time

**Features**:
- Each team = one line (team colors)
- League average = bold accent line
- Rolling 3-month average (smooth)
- X-axis: Time (years)
- Y-axis: ELO rating

**Interpretation**:
- See league-wide trends
- Identify dynasties (sustained high ELO)
- Spot team trajectories (improving/declining)
- Compare teams visually

### 2. Team Small Multiples (`plot_team_small_multiples`)
**Output**: `Outputs/team_small_multiples.png`

**Visualization**: Grid of 32 subplots, one per team

**Features**:
- 8 columns × 4 rows
- Each subplot = one team's ELO over time
- Team colors for lines
- Compact year labels (2-digit, rotated)
- Rolling averages

**Interpretation**:
- Individual team trajectories
- Compare teams side-by-side
- Identify patterns (rebuilds, dynasties, volatility)

### 3. Season Ladder (`plot_season_ladder`)
**Output**: `Outputs/season_ladder_[season].png`

**Visualization**: Horizontal bar chart of end-of-season rankings

**Features**:
- Bars sorted by ELO (highest to lowest)
- Team colors for bars
- ELO values labeled
- One plot per season (default: 2024)

**Interpretation**:
- Final season standings
- ELO-based rankings
- Visual hierarchy of teams

### 4. Team Summary Statistics (`plot_team_summary_stats`)
**Output**: `Outputs/team_summary_stats.png`

**Visualization**: Multi-panel dashboard

**Panels**:
1. **Win Percentage vs ELO Rating**: Scatter plot
   - X-axis: ELO rating
   - Y-axis: Win percentage
   - Team names toggleable
2. **ELO Distribution**: Histogram
3. **Peak vs Current ELO**: Scatter plot
4. **ELO Change Over Time**: Line plot

**Interpretation**:
- Overall team performance metrics
- Relationship between ELO and wins
- Historical peaks vs current state

### 5. Biggest Swings (`plot_biggest_swings`)
**Output**: `Outputs/biggest_swings.png`

**Visualization**: Table showing top 10 upsets

**Columns**:
- Date, Team, Opponent
- ELO Change (positive = gain, negative = loss)
- Score (Team-Opponent)
- Outcome

**Interpretation**:
- Most impactful games in NFL history
- Biggest underdog wins
- Biggest favorite losses

### 6. Parity Metrics (`plot_parity_metrics`)
**Output**: `Outputs/parity_metrics.png`

**Visualization**: Multi-line plot showing parity components over time

**Lines**:
- Dispersion (STD)
- Rank Stability (Spearman)
- Upset Gap
- Forecast Entropy

**Interpretation**:
- How league parity has changed over time
- Which components drive parity
- Trends in competitiveness

### 7. Combined Parity (`plot_combined_parity`)
**Output**: `Outputs/combined_parity.png`

**Visualization**: Line plot of composite Parity Index

**Features**:
- Single line showing Parity Index over time
- Higher = more parity
- Lower = less parity

**Interpretation**:
- Overall league competitiveness trend
- Identify most/least competitive seasons
- Compare eras

### 8. Season Delta Normalized (`plot_season_delta_normalized`)
**Output**: `Outputs/delta_normalized_[season].png`

**Visualization**: Horizontal bar chart of normalized ELO changes

**Features**:
- Bars show z-score of ELO change
- Positive = improved more than average
- Negative = declined more than average
- Team colors

**Interpretation**:
- Which teams improved/declined most
- Relative to league average
- Normalized for cross-season comparison

### 9. Conference Balance (`plot_conference_balance`)
**Output**: `Outputs/conference_balance.png`

**Visualization**: Line plot showing AFC vs NFC average ELO

**Features**:
- One line (blue) showing difference
- Positive = AFC stronger
- Negative = NFC stronger

**Interpretation**:
- Conference dominance over time
- Balance trends
- Which conference is stronger

### 10. Luck Index (`plot_luck_index`)
**Output**: `Outputs/luck_index_[season].png`

**Visualization**: Lollipop plot of Luck Index by team

**Features**:
- Horizontal bars (team colors)
- Positive = lucky (over-performed)
- Negative = unlucky (under-performed)
- Top N teams shown (default: 20)

**Interpretation**:
- Which teams got lucky/unlucky
- Over/under-performance vs expectations
- Clutch factor analysis

### 11. Reliability Curve (`plot_calibration_and_brier`)
**Output**: `Outputs/reliability_curve.png`

**Visualization**: Scatter plot of predicted vs actual win rates

**Features**:
- X-axis: Mean predicted win probability (by bin)
- Y-axis: Mean actual outcome (by bin)
- Perfect calibration line (y=x)
- Points = bins of games

**Interpretation**:
- Model calibration quality
- Over/under-confidence
- Forecast accuracy

### 12. Any Given Sunday Meter (`plot_any_given_sunday_line`)
**Output**: `Outputs/any_given_sunday_meter.png`

**Visualization**: Line plot of upset gap over time

**Features**:
- X-axis: Season
- Y-axis: Gap (Actual Upsets - Expected Upsets)
- Positive = more chaos than expected
- Negative = favorites win more than expected

**Interpretation**:
- "Any Given Sunday" effect over time
- League unpredictability
- Trends in upset frequency

### 13. Luck Timelines (`plot_luck_timelines`)
**Output**: `Outputs/luck_timelines.png`

**Visualization**: Line plot showing team luck over time

**Features**:
- Multiple lines (one per team, team colors)
- League average line (accent color)
- Rolling averages
- X-axis: Season
- Y-axis: Luck Index

**Interpretation**:
- Team luck trends over time
- Sustained luck vs random variation
- League-wide luck patterns

### 14. SoS vs Final ELO (`plot_sos_vs_elo`)
**Output**: `Outputs/sos_vs_elo_[season].png`

**Visualization**: Scatter plot

**Features**:
- X-axis: Strength of Schedule (configurable: SoS_PreMean, SoS_EndMean, SoS_PreWeighted, SoS_EndWeighted)
- Y-axis: End-of-Season ELO
- Team colors for points
- Quadrant lines (optional)

**Configuration**:
```python
config.PLOT_CONFIG['sos']['sos_metric'] = 'SoS_PreMean'  # Default
# Options: 'SoS_PreMean', 'SoS_EndMean', 'SoS_PreWeighted', 'SoS_EndWeighted'
```

**Interpretation**:
- Schedule difficulty vs performance
- Teams that succeeded despite hard schedules
- Teams that failed despite easy schedules
- Different SoS metrics show different perspectives (pre-game vs end-of-season, weighted vs unweighted)

### 15. Team Story Card (`export_team_story_card`)
**Output**: `Outputs/team_story_card.png`

**Visualization**: One-page summary dashboard

**Sections**:
- Team name and season
- Key Performance Indicators (Wins, ELO, Luck Index)
- Biggest swing in season
- ELO progression sparkline

**Interpretation**:
- Quick team summary
- Season highlights
- Performance snapshot

---

## Output Interpretations

### ELO Ratings
- **Range**: Typically 800-1800
- **1000**: Average team (starting point)
- **1200+**: Above average
- **1400+**: Very good
- **1600+**: Elite
- **800-**: Very poor

### Luck Index
- **+2 to +3**: Very lucky season (won 2-3 more games than expected)
- **+1**: Somewhat lucky
- **0**: Performed as expected
- **-1**: Somewhat unlucky
- **-2 to -3**: Very unlucky season

### Parity Index
- **+1.0+**: Very high parity (competitive, unpredictable)
- **+0.5**: High parity
- **0**: Average parity
- **-0.5**: Low parity
- **-1.0-**: Very low parity (predictable, hierarchical)

### Calibration
- **On y=x line**: Perfect calibration
- **Below line**: Model overconfident (predicted too high)
- **Above line**: Model underconfident (predicted too low)
- **Brier < 0.20**: Good forecast accuracy
- **Brier > 0.25**: Poor forecast accuracy

### Upset Gap
- **+0.10**: 10% more upsets than expected (chaotic)
- **0**: Expected number of upsets
- **-0.10**: 10% fewer upsets than expected (predictable)

---

## Usage Guide

### Basic Usage
```python
python NFL_ELO_organized.py
```

### Customizing Configuration

#### Change ELO Parameters
```python
config.START_ELO = 1500.0  # Different starting point
config.K_FACTOR = 30.0     # More volatile ratings (default is 20.0)
config.HOME_FIELD_ADVANTAGE = 70.0  # Stronger home advantage
```

#### Toggle Plots
```python
config.toggle_plot('league_tapestry')  # Toggle specific plot
config.disable_all_plots()             # Disable all
config.enable_all_plots()              # Enable all
```

#### Change Plot Settings
```python
config.PLOT_CONFIG['season_ladder']['season'] = 2023  # Different season
config.PLOT_CONFIG['team_summary']['show_team_names'] = False  # Hide names
```

#### Enable/Disable Display
```python
config.SHOW_PLOTS = True   # Show plots (slower)
config.SHOW_PLOTS = False  # Save only (faster)
```

### Programmatic Usage

#### Load Data
```python
raw_data = load_nfl_data()
elo_games = calculate_elo_ratings(raw_data)
timeline = create_team_timeline(elo_games)
```

#### Calculate Metrics
```python
luck_index = build_luck_index(timeline)
calibration_curve, perf = build_calibration_bins(elo_games)
parity_metrics = build_parity_metrics_clean(season_team_summary, elo_games)
```

#### Generate Specific Plots
```python
plot_league_tapestry(team_monthly_elo)
plot_luck_index(luck_index, season=2024)
plot_sos_vs_elo(sos_data, season=2024)
```

### Output Files
All plots are saved to `Outputs/` directory:
- `league_tapestry.png`
- `team_small_multiples.png`
- `season_ladder_2024.png`
- `luck_index_2024.png`
- `reliability_curve.png`
- `any_given_sunday_meter.png`
- And more...

### Troubleshooting

#### File Not Found
- Ensure `NFLELO_data.xlsx` is in the same directory
- Check file path in `config.DATA_FILE`

#### Memory Issues
- Disable plots you don't need: `config.disable_all_plots()`
- Then enable only what you need: `config.toggle_plot('plot_name', True)`
- Set `config.SHOW_PLOTS = False` to avoid displaying plots

#### Plot Errors
- Check that required data structures exist
- Verify plot is enabled: `config.is_plot_enabled('plot_name')`
- Check plot configuration: `config.get_plot_config('plot_name')`

---

## Extending the Code

### Adding a New Plot
1. Create plotting function with `@safefig` decorator
2. Add plot toggle to `config.PLOTS_ENABLED`
3. Add plot config to `config.PLOT_CONFIG`
4. Call function in `main()`

### Adding a New Metric
1. Create calculation function
2. Return DataFrame with metric
3. Optionally create plotting function
4. Integrate into `main()`

### Modifying ELO Calculation
- Edit `calculate_elo_ratings()` function
- Adjust formulas for expected win probability
- Modify K-factor or home field advantage logic

### Customizing Colors
- Edit `config.COLORS` dictionary
- Add team colors to `config.COLORS['team_colors']`
- Use `config.apply_color_scheme(ax)` in plots

---

## Mathematical Formulas Reference

### ELO Expected Win Probability
```
Ea = 1 / (1 + 10^((Rb_eff - Ra_eff) / 400))
```

### ELO Update
```
Delta = K * (Actual - Expected)
New_Rating = Old_Rating + Delta
```

### Brier Score
```
Brier = mean((Actual - Predicted)^2)
```

### Log Loss
```
LogLoss = -mean(Actual * log(Predicted) + (1-Actual) * log(1-Predicted))
```

### Entropy
```
Entropy = -p*log(p) - (1-p)*log(1-p)
```

### Z-Score
```
Z = (X - mean(X)) / std(X)
```

### Spearman Correlation
```
RankStability = corr(Rank_Start, Rank_End, method='spearman')
```

---

## Data Flow Summary

```
Excel File (NFLELO_data.xlsx)
    ↓
load_nfl_data()
    ↓
Raw DataFrame (df)
    ↓
calculate_elo_ratings()
    ↓
ELO Games (elo_games)
    ├─→ create_team_timeline() → Timeline (timeline)
    │       ├─→ create_monthly_elo_data() → Monthly ELO (team_monthly_elo)
    │       ├─→ build_luck_index() → Luck Index (luck_df)
    │       ├─→ create_season_team_summary() → Season Summary (season_team_summary)
    │       │       ├─→ create_season_conference_summary() → Conference Summary
    │       │       ├─→ build_sos() → SoS Data
    │       │       └─→ build_parity_metrics_clean() → Parity Metrics
    │       └─→ create_elo_games_data() → ELO Games (for calibration)
    │               ├─→ build_calibration_bins() → Calibration Data
    │               └─→ build_upset_map() → Upset Map
    ↓
Visualizations (15+ plots)
    ↓
Output Files (PNG in Outputs/)
```

---

## Recent Updates and Enhancements

### Major Features Added (2025)

1. **Real Strength of Schedule**: Implemented 4 SoS metrics (PreMean, EndMean, PreWeighted, EndWeighted) using actual opponent ELO ratings

2. **Season Reset/Regression**: Added configurable season boundary regression to prevent runaway ELO drift

3. **Margin of Victory Scaling**: Implemented MOV-based K-factor scaling with multiple methods (elo_standard, log, sqrt)

4. **Segmented Calibration**: Added ability to segment calibration analysis by era, ELO difference, and home/away

5. **Close-Game Luck Index**: Added close-game specific luck metrics to separate coin-flip games from model mis-rating

6. **Pythagorean Expectation**: Added Pythagorean expected wins and luck metrics for comparison with ELO-based luck

7. **Rescaled Parity Index**: Added 0-100 Parity Score anchored to reference era for more intuitive interpretation

8. **ELO Volatility & Dynasty Score**: Added metrics to measure team consistency and sustained dominance

9. **Division-Level Analysis**: Added division summaries with parity metrics and strength rankings

10. **Forecast Sharpness**: Added metric to evaluate if model makes strong calls or is too timid

11. **League Mean Tracking**: Added optional tracking of league-wide mean ELO over time

### Configuration Updates
All new features are fully configurable via `NFLConfig` class parameters. See [Configuration System](#configuration-system) for details.

### Data Structure Updates
New columns and DataFrames have been added. See [Data Structures](#data-structures) for complete details.

---

## Conclusion

This documentation provides a comprehensive guide to the NFL ELO Analysis Tool. The system is designed to be:
- **Modular**: Each function is independent and reusable
- **Configurable**: Extensive customization options
- **Extensible**: Easy to add new analyses and visualizations
- **Robust**: Error handling and memory management
- **Comprehensive**: Covers multiple aspects of NFL analysis
- **Advanced**: Includes sophisticated metrics like MOV scaling, segmented calibration, and multiple luck indices

For questions or extensions, refer to the code comments and function docstrings for additional details.

