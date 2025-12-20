"""
NFL ELO Rating System Analysis Tool

A comprehensive tool for analyzing NFL team performance using ELO ratings.
Provides various visualizations and statistical analyses of team performance
over time, including league-wide trends, individual team trajectories,
and advanced metrics like parity analysis and luck indices.

Features:
- ELO rating calculations with home field advantage
- League-wide tapestry showing all teams over time
- Individual team trajectory analysis
- Season ladder rankings
- Team summary statistics
- Biggest ELO swings analysis
- League parity metrics over time
- Configurable plot toggles for performance
- Safe figure handling with automatic cleanup

Usage:
    python NFL_ELO_organized.py

Configuration:
    - Modify the NFLConfig class to adjust ELO parameters
    - Use config.toggle_plot() to enable/disable specific plots
    - Use config.disable_all_plots() for quick analysis without visualizations

Data Requirements:
    - NFLELO_data.xlsx file in the same directory
    - Excel file should contain columns: Date, Team, Opp, Result, Hosting

Author: Walker Tracy
Created: October 17th, 2025
"""

# =============================================================================
# IMPORTS AND SETUP
# =============================================================================

# Standard library imports
import os
import sys
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

# Third-party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns

# Configure pandas display options
pd.options.display.max_columns = None
pd.options.display.width = None

# Configure matplotlib
plt.style.use('default')
sns.set_palette("husl")

# =============================================================================
# CONFIGURATION SYSTEM
# =============================================================================

class NFLConfig:
    """Configuration class for NFL ELO analysis with plot toggles."""
    
    def __init__(self):
        # Core ELO parameters
        self.START_ELO = 1000.0
        self.K_FACTOR = 20.0
        self.HOME_FIELD_ADVANTAGE = 55.0
        self.ROLLING_WINDOW_MONTHS = 3
        
        # Season reset parameters
        self.SEASON_RESET_LAMBDA = 0.15  # Regression to mean factor (0.0 = no reset, 1.0 = full reset)
        self.USE_SEASON_RESET = True  # Enable season reset/regression
        
        # Margin of victory parameters
        self.USE_MARGIN_OF_VICTORY = True  # Enable MOV scaling
        self.MOV_SCALING_METHOD = "elo_standard"  # "elo_standard", "log", "sqrt"
        
        # League mean anchoring
        self.TRACK_LEAGUE_MEAN = True  # Track league-wide mean ELO
        self.RECENTER_ELO = False  # Re-center ELO to START_ELO after each game (not recommended)
        
        # Parity index weights
        self.PARITY_WEIGHTS = {
            "dispersion": 0.35,
            "rank_stab": 0.25,
            "upset_gap": 0.25,
            "entropy": 0.15,
        }
        
        # Parity index scaling
        self.PARITY_ANCHOR_ERA_START = 1970  # Start era for parity scaling
        self.PARITY_ANCHOR_ERA_END = 1979  # End era for parity scaling
        self.PARITY_SCALE_MIN = 0  # Minimum parity score
        self.PARITY_SCALE_MAX = 100  # Maximum parity score
        
        # Pythagorean expectation
        self.PYTHAG_EXPONENT = 2.37  # NFL exponent for Pythagorean expectation
        
        # Dynasty score threshold
        self.DYNASTY_THRESHOLD = 1150.0  # ELO threshold for dynasty score calculation
        
        # Close game definition
        self.CLOSE_GAME_POINT_DIFF = 7  # Point differential threshold
        self.CLOSE_GAME_PROB_RANGE = (0.45, 0.55)  # Win probability range for close games
        
        # Custom color scheme - NFL-inspired with modern aesthetics
        self.COLORS = {
            # Primary palette - Deep, rich colors
            'primary': '#1a365d',      # Deep navy blue
            'secondary': '#2d3748',    # Charcoal gray
            'accent': '#e53e3e',       # NFL red
            'success': '#38a169',      # Forest green
            'warning': '#d69e2e',      # Golden yellow
            'info': '#3182ce',         # Bright blue
            
            # Extended palette for multiple series
            'series': [
                '#1a365d',  # Deep navy
                '#e53e3e',  # NFL red
                '#38a169',  # Forest green
                '#d69e2e',  # Golden yellow
                '#805ad5',  # Purple
                '#dd6b20',  # Orange
                '#319795',  # Teal
                '#c53030',  # Dark red
                '#2f855a',  # Dark green
                '#b7791f',  # Dark yellow
                '#553c9a',  # Dark purple
                '#c05621',  # Dark orange
            ],
            
            # Gradient colors for heatmaps
            'heatmap_low': '#f7fafc',   # Light gray
            'heatmap_high': '#1a365d',  # Deep navy
            
            # Team colors (NFL-inspired)
            'team_colors': {
                'ARI': '#97233f', 'ATL': '#a71930', 'BAL': '#241773', 'BUF': '#00338d',
                'CAR': '#0085ca', 'CHI': '#0b162a', 'CIN': '#fb4f14', 'CLE': '#311d00',
                'DAL': '#003594', 'DEN': '#fb4f14', 'DET': '#0076b6', 'GNB': '#203731',
                'HOU': '#03202f', 'IND': '#002c5f', 'JAX': '#006778', 'KAN': '#e31837',
                'LVR': '#000000', 'LAC': '#0080c6', 'LAR': '#003594', 'MIA': '#008e97',
                'MIN': '#4f2683', 'NWE': '#002244', 'NOR': '#d3bc8d', 'NYG': '#0b2265',
                'NYJ': '#125740', 'PHI': '#004c54', 'PIT': '#ffb612', 'SFO': '#aa0000',
                'SEA': '#002244', 'TAM': '#d50a0a', 'TEN': '#0c2340', 'WAS': '#5a1414'
            },
            
            # Background and text colors
            'background': '#ffffff',
            'text_primary': '#1a202c',
            'text_secondary': '#4a5568',
            'grid': '#e2e8f0',
            'border': '#cbd5e0'
        }
        
        # Plot toggles - set to False to skip time-consuming plots
        self.PLOTS_ENABLED = {
            'league_tapestry': True,
            'team_small_multiples': True,
            'season_ladder': True,
            'team_summary': True,
            'biggest_swings': True,
            'parity_metrics': True,
            'combined_parity': True,
            'delta_normalized': True,
            'division_heatmap': True,
            'conference_balance': True,
            'parity_components': True,
            'parity_index': True,
            'luck_index': True,
            'calibration_plots': True,
            'upset_heatmap': True,
            'swing_hall_of_fame': True,
            'volatility_bars': True,
            'luck_timelines': True,
            'luck_small_multiples': True,
            'subway_map': True,
            'story_card': True,
            'franchise_poster': True,
            'any_given_sunday': True,
            'sos_vs_elo': True
        }
        
        # Plot configurations
        self.PLOT_CONFIG = {
            'league_tapestry': {
                'use_rolling': True,
                'window': 3,
                'figsize': (12, 7),
                'line_alpha': 0.25,
                'team_linewidth': 0.8,
                'league_linewidth': 2.5,
            },
            'team_small_multiples': {
                'teams': None,  # None = all 32 teams
                'use_rolling': True,
                'window': 3,
                'cols': 8,
                'show_baseline': False,
                'baseline_value': 1500.0,
                'show_realignment': False,
                'realignment_date': "2002-09-01",
                'fig_cell_w': 2.2,
                'fig_cell_h': 1.7,
            },
            'season_ladder': {
                'season': 2024,
                'top_n': None,
            },
            'team_summary': {
                'show_team_names': True,  # Toggle for team names on scatter plot
                'top_n': 16,
            },
            'delta_normalized': {
                'season': 2024,
                'top_n': None,
            },
            'division_heatmap': {
                'conference': None,  # "AFC", "NFC", or None
            },
            'conference_balance': {
                'figsize': (11, 5),
            },
            'parity': {
                'use_dispersion': True,
                'use_rank_stability': True,
                'use_upset_gap': True,
                'use_entropy': True,
                'w_dispersion': 0.35,
                'w_rank_stability': 0.25,
                'w_upset_gap': 0.25,
                'w_entropy': 0.15,
                'smooth_ma': 1,
                'upset_definition': "elo_pre",
            },
            'parity_plots': {
                'show_components': True,
                'figsize_components': (12, 6),
                'figsize_index': (11, 5),
                'legend_cols': 2,
            },
            'rebounds': {
                'n_games_threshold': 6,
                'low_cut': 1500.0,
                'finish_cut': 1600.0,
            },
            'dynasties': {
                'threshold': 1650.0,
                'use_col': 'Elo_RollMed',
            },
            'changepoints': {
                'k': 7,
            },
            'luck_index': {
                'figsize': (12, 6),
                'top_n': 20,
            },
            'calibration': {
                'figsize': (12, 5),
                'n_bins': 10,
            },
            'upset_heatmap': {
                'figsize': (10, 8),
            },
            'swing_hof': {
                'top_n': 25,
                'figsize': (12, 8),
            },
            'volatility': {
                'figsize': (12, 6),
                'top_n': 20,
            },
            'luck_timelines': {
                'figsize': (14, 8),
                'teams': None,
            },
            'luck_small_multiples': {
                'figsize_cell': (2.5, 1.8),
                'cols': 8,
            },
            'presentation': {
                'subway': {
                    'season': 2024,
                    'label_top_k_swings': 3,
                    'week_ending': None,
                },
                'story_card': {
                    'season': 2024,
                    'team': None,
                    'export_path': 'team_story_card.png',
                },
                'poster': {
                    'team': 'KAN',
                    'roll_window': 3,
                    'annotate_dynasties': True,
                    'export_path': 'franchise_timeline_poster.png',
                }
            },
            'sos': {
                'season': None,
                'weighting': 'equal',
                'top_n': None,
                'show_quadrants': True
            }
        }
        
        # File paths
        self.DATA_FILE = 'NFLELO_data.xlsx'
        
        # Output directory - create "Outputs" folder
        self.OUTPUT_DIR = Path("Outputs")
        self.OUTPUT_DIR.mkdir(exist_ok=True)  # Create the folder if it doesn't exist
        
        # Display settings
        self.SHOW_PLOTS = False  # Set to False to disable plot display (plots still saved)
        
    def is_plot_enabled(self, plot_name: str) -> bool:
        """Check if a specific plot is enabled."""
        return self.PLOTS_ENABLED.get(plot_name, False)
    
    def get_plot_config(self, plot_name: str) -> Dict:
        """Get configuration for a specific plot."""
        return self.PLOT_CONFIG.get(plot_name, {})
    
    def apply_color_scheme(self, ax=None, style='default'):
        """Apply the custom color scheme to a plot."""
        if ax is None:
            ax = plt.gca()
        
        # Set background and grid colors
        ax.set_facecolor(self.COLORS['background'])
        ax.grid(True, color=self.COLORS['grid'], alpha=0.3, linewidth=0.5)
        
        # Set text colors
        ax.tick_params(colors=self.COLORS['text_primary'])
        ax.xaxis.label.set_color(self.COLORS['text_primary'])
        ax.yaxis.label.set_color(self.COLORS['text_primary'])
        ax.title.set_color(self.COLORS['text_primary'])
        
        # Set spine colors
        for spine in ax.spines.values():
            spine.set_color(self.COLORS['border'])
            spine.set_linewidth(0.8)
        
        return ax
    
    def get_team_color(self, team: str) -> str:
        """Get the official team color for a given team."""
        return self.COLORS['team_colors'].get(team, self.COLORS['primary'])
    
    def get_series_color(self, index: int) -> str:
        """Get a color from the series palette by index."""
        return self.COLORS['series'][index % len(self.COLORS['series'])]
    
    def toggle_plot(self, plot_name: str, enabled: bool = None) -> None:
        """Toggle a specific plot on/off."""
        if enabled is None:
            self.PLOTS_ENABLED[plot_name] = not self.PLOTS_ENABLED.get(plot_name, False)
        else:
            self.PLOTS_ENABLED[plot_name] = enabled
    
    def disable_all_plots(self) -> None:
        """Disable all plots for quick analysis."""
        for plot_name in self.PLOTS_ENABLED:
            self.PLOTS_ENABLED[plot_name] = False
    
    def enable_all_plots(self) -> None:
        """Enable all plots."""
        for plot_name in self.PLOTS_ENABLED:
            self.PLOTS_ENABLED[plot_name] = True

# Global configuration instance
config = NFLConfig()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safefig(func):
    """
    Decorator to safely handle figure creation and saving.
    
    This wrapper ensures that matplotlib figures are properly closed even if
    an error occurs. This prevents memory leaks from accumulating unclosed figures.
    Think of it as a safety net for plotting functions.
    
    Usage:
        @safefig
        def my_plotting_function():
            fig, ax = plt.subplots()
            # ... plotting code ...
    """
    def wrapper(*args, **kwargs):
        figure = None
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as error:
            print(f"[ERROR] {func.__name__}: {error}")
            if figure:
                plt.close(figure)
            raise
        finally:
            if figure:
                plt.close(figure)
    return wrapper

def season_from_date(date: pd.Timestamp) -> Union[int, float]:
    """
    Convert a date to the NFL season year.
    
    NFL seasons are weird - they span two calendar years!
    - Games in January/February belong to the PREVIOUS season
    - Games in August through December belong to the CURRENT calendar year
    
    Examples:
        - January 15, 2024 → Season 2023 (it's the playoffs from the 2023 season)
        - September 10, 2024 → Season 2024 (it's the regular season)
        - February 5, 2024 → Season 2023 (Super Bowl from 2023 season)
    
    Args:
        date: The date to convert
        
    Returns:
        The season year (e.g., 2024) or NaN if the date is invalid
    """
    if pd.isna(date):
        return np.nan
    
    date = pd.Timestamp(date)
    
    # If it's January or February, it's part of the previous year's season
    if date.month in (1, 2):
        return date.year - 1
    else:
        # August through December belong to the current year's season
        return date.year

def canonical_franchise(team_code: str, game_date: pd.Timestamp) -> str:
    """
    Map historical team codes to current 32 NFL franchises.
    
    Teams have moved around a lot! The Rams were in LA, then St. Louis, then back to LA.
    The Raiders were in Oakland, then LA, then back to Oakland, then Las Vegas.
    This function handles all those relocations and name changes so we can track
    franchises consistently over time.
    
    Args:
        team_code: The team code from the data (e.g., "STL", "OAK", "SDG")
        game_date: The date of the game (needed to figure out which city/name was active)
        
    Returns:
        The canonical franchise code (e.g., "LAR", "LVR", "LAC")
    """
    # Clean up the input
    team_code = (team_code or "").strip().upper()
    
    # Static mappings for relocations
    static_map = {
        "SDG": "LAC", "LAC": "LAC",
        "OAK": "LVR", "RAI": "LVR", "LVR": "LVR", 
        "RAM": "LAR",  # Note: LAR and STL handled separately with date logic
        "PHO": "ARI", "ARI": "ARI",
        "BOS": "NWE", "NWE": "NWE",
        "OTI": "TEN",
    }
    
    # Check if this team code has a simple static mapping (no date logic needed)
    if team_code in static_map:
        return static_map[team_code]
    
    # Handle Los Angeles - this is tricky because both Rams and Raiders were there!
    # Los Angeles Rams: 1946-1994 (LA), 1995-2015 (St. Louis), 2016+ (back to LA)
    # Los Angeles Raiders: 1982-1994 (LA), then back to Oakland, then Las Vegas
    if team_code == "LAR":
        # Between 1982-1994, "LAR" could mean the Raiders (who were in LA then)
        if game_date >= pd.Timestamp("1982-01-01") and game_date < pd.Timestamp("1995-01-01"):
            return "LVR"  # Los Angeles Raiders → Las Vegas Raiders
        # All other times, "LAR" means the Los Angeles Rams
        else:
            return "LAR"
    
    # Handle St. Louis - Cardinals were there first, then Rams moved in
    # St. Louis Cardinals: 1960-1987, then moved to Phoenix (became Arizona)
    # St. Louis Rams: 1995-2015, then moved back to LA
    if team_code == "STL":
        # Before 1995, "STL" means the Cardinals
        if game_date < pd.Timestamp("1995-01-01"):
            return "ARI"  # St. Louis Cardinals → Arizona Cardinals
        # 1995-2015, "STL" means the Rams
        else:
            return "LAR"  # St. Louis Rams → Los Angeles Rams
    
    # Handle Houston - the Oilers became the Titans, then Houston got a new team
    # Houston Oilers: pre-1999 → became Tennessee Titans
    # Houston Texans: 1999+ → new expansion team
    if team_code == "HOU":
        if game_date < pd.Timestamp("1999-01-01"):
            return "TEN"  # Houston Oilers → Tennessee Titans
        else:
            return "HOU"  # Houston Texans (new team)
    
    # Handle Baltimore - Colts left, then Ravens arrived
    # Baltimore Colts: pre-1984 → moved to Indianapolis
    # Baltimore Ravens: 1996+ → new team (from Browns relocation)
    if team_code == "BAL":
        if game_date < pd.Timestamp("1984-01-01"):
            return "IND"  # Baltimore Colts → Indianapolis Colts
        elif game_date < pd.Timestamp("1996-01-01"):
            # Gap period (1984-1995) - no team in Baltimore
            # If we see BAL here, it's probably an error, but map to IND for continuity
            return "IND"
        else:
            return "BAL"  # Baltimore Ravens (new team in 1996)
    
    # Handle Dallas - Cowboys vs Chiefs confusion
    # Dallas Cowboys: 1960+ (always in Dallas)
    # Dallas Texans: 1960-1962 (became Kansas City Chiefs in 1963)
    # Since our data starts in 1970, DAL is always the Cowboys
    if team_code == "DAL":
        return "DAL"  # No change needed for 1970+ data
    
    # If we don't recognize the code, just return it as-is
    # (might be a valid current team code)
    return team_code

def _z(x: pd.Series) -> pd.Series:
    """Calculate z-scores for a series."""
    return (x - x.mean()) / x.std()

def _moving_avg(x: pd.Series, k: int) -> pd.Series:
    """Calculate moving average with window k."""
    return x.rolling(window=k, min_periods=1).mean()

def add_rolling_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Add rolling statistical features to a DataFrame.
    
    For monthly ELO data, this applies rolling features per franchise.
    For timeline data, this applies rolling features per team.
    
    Args:
        df: DataFrame with numeric columns
        window: Rolling window size
        
    Returns:
        DataFrame with additional rolling features
    """
    df = df.copy()
    
    # Determine the grouping column (Franchise for monthly data, Team for timeline)
    group_col = 'Franchise' if 'Franchise' in df.columns else 'Team'
    
    # Sort by group and time
    time_col = 'YearMonth' if 'YearMonth' in df.columns else 'Date'
    df = df.sort_values([group_col, time_col]).reset_index(drop=True)
    
    # Apply rolling features per group
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in ['Year', 'Month', 'YearMonth', 'Date']:
            df[f'{col}_RollMean'] = (
                df.groupby(group_col)[col]
                .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
            )
            df[f'{col}_RollMed'] = (
                df.groupby(group_col)[col]
                .transform(lambda s: s.rolling(window=window, min_periods=1).median())
            )
            df[f'{col}_RollStd'] = (
                df.groupby(group_col)[col]
                .transform(lambda s: s.rolling(window=window, min_periods=1).std())
            )
    
    return df

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_nfl_data(file_path: str = None) -> pd.DataFrame:
    """
    Load and preprocess NFL ELO data from Excel file.
    
    Args:
        file_path: Path to Excel file (defaults to config.DATA_FILE)
        
    Returns:
        Loaded and preprocessed DataFrame
    """
    if file_path is None:
        file_path = config.DATA_FILE
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    print(f"Loading NFL data from {file_path}...")
    
    # Load the data
    df = pd.read_excel(file_path)
    
    # Clean up columns
    df = df.drop(columns=["Day", "Game Number", "Week"], errors="ignore")
    df["Result"] = df["Result"].str.replace(" (OT)", "", regex=False)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y", errors="coerce")
    df = df[df["Date"] <= "2025-07-01"].copy()
    
    # Add GameID if not present
    if "GameID" not in df.columns:
        df = df.sort_values(["Date", "Team", "Opp"]).reset_index(drop=True)
        df["GameID"] = np.arange(1, len(df) + 1)
    
    # Parse game results
    extracted = df["Result"].str.strip().str.extract(r"^([WLT])\s+(\d+)-(\d+)$")
    df["Outcome"] = extracted[0]
    df["TeamPoints"] = pd.to_numeric(extracted[1], errors="coerce")
    df["OppPoints"] = pd.to_numeric(extracted[2], errors="coerce")
    df["PointDiff"] = df["TeamPoints"] - df["OppPoints"]
    df["Win"] = (df["Outcome"] == "W").astype("Int64")
    df["Loss"] = (df["Outcome"] == "L").astype("Int64")
    df["Tie"] = (df["Outcome"] == "T").astype("Int64")
    
    # Handle home field advantage
    host_raw = df["Hosting"].fillna("").astype(str).str.strip()
    is_neutral = host_raw.str.upper().eq("N")
    A_Home = np.where(is_neutral, 0, np.where(host_raw.eq("@"), 0, 1))
    B_Home = np.where(is_neutral, 0, 1 - A_Home)
    df["A_Home"] = A_Home
    df["B_Home"] = B_Home
    
    # Add canonical franchise mapping
    df["Team_FR"] = [
        canonical_franchise(code, dt) for code, dt in zip(df["Team"], df["Date"])
    ]
    df["Opp_FR"] = [
        canonical_franchise(code, dt) for code, dt in zip(df["Opp"], df["Date"])
    ]
    
    # Add season
    df["Season"] = df["Date"].map(season_from_date).astype("Int64")
    
    print(f"Loaded {len(df)} records from {df['Date'].min().year} to {df['Date'].max().year}")
    
    return df

def calculate_elo_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate ELO ratings for all teams based on game results.
    
    This is the heart of the ELO system - it processes every game and updates
    team ratings based on outcomes. Think of it like a chess rating system
    adapted for football, where beating a strong team gives you more points
    than beating a weak team.
    
    Features:
    - Margin of victory scaling (blowouts matter more than close wins)
    - Season reset/regression to mean (prevents ratings from drifting too far)
    - League mean tracking (monitors overall league strength over time)
    
    Args:
        df: DataFrame with game data (each game appears twice - once per team)
        
    Returns:
        DataFrame with ELO calculations for every game
    """
    print("Calculating ELO ratings...")
    
    # Step 1: Prepare the game data
    # Our data has each game listed twice (once for each team's perspective)
    # We need to deduplicate so we only process each game once
    print("  Preparing game data...")
    games_prep = (
        df.rename(columns={"Team_FR": "TeamA", "Opp_FR": "TeamB"})
        .sort_values(["Date", "GameID"])[
            ["Date", "GameID", "Season", "TeamA", "TeamB", "A_Home", "B_Home",
             "TeamPoints", "OppPoints", "Outcome"]
        ]
        .reset_index(drop=True)
    )
    
    # Step 2: Remove duplicate games
    # Create a unique key for each game by combining date and team names
    # This way we can identify and remove the duplicate entries
    def create_game_key(row):
        """Create a unique identifier for each game."""
        teams_sorted = sorted([row['TeamA'], row['TeamB']])
        return f"{row['Date']}_{teams_sorted[0]}_{teams_sorted[1]}"
    
    games_prep['GameKey'] = games_prep.apply(create_game_key, axis=1)
    original_row_count = len(games_prep)
    
    # Keep only the first occurrence of each unique game
    unique_games = games_prep.drop_duplicates(subset=['GameKey'], keep='first').drop(columns=['GameKey'])
    
    # Step 3: Verify data quality
    # Make sure all teams appear in both columns (this catches data issues)
    teams_as_team_a = set(unique_games['TeamA'].unique())
    teams_as_team_b = set(unique_games['TeamB'].unique())
    all_unique_teams = teams_as_team_a.union(teams_as_team_b)
    
    teams_only_in_a = teams_as_team_a - teams_as_team_b
    teams_only_in_b = teams_as_team_b - teams_as_team_a
    
    if teams_only_in_a or teams_only_in_b:
        print(f"  [WARNING] Some teams only appear in one column (possible data issue):")
        if teams_only_in_a:
            print(f"           Only in TeamA column: {sorted(teams_only_in_a)}")
        if teams_only_in_b:
            print(f"           Only in TeamB column: {sorted(teams_only_in_b)}")
    
    print(f"  [OK] Deduplicated: {original_row_count:,} rows → {len(unique_games):,} unique games")
    print(f"  [OK] Found {len(all_unique_teams)} unique teams")
    
    # Step 4: Initialize the rating system
    # Start all teams at the baseline ELO rating
    current_ratings = {}  # Dictionary: team_name -> current_elo_rating
    ratings_at_end_of_season = {}  # Track ratings at end of each season
    league_average_over_time = []  # Track how the league average changes
    
    def get_current_rating(team_name: str) -> float:
        """Get a team's current ELO rating, or default to starting value if new team."""
        return current_ratings.get(team_name, config.START_ELO)
    
    def calculate_margin_of_victory_factor(point_difference: float, 
                                         team_a_effective_elo: float, 
                                         team_b_effective_elo: float) -> float:
        """
        Calculate how much the margin of victory should affect the rating change.
        
        A 30-point blowout should matter more than a 3-point squeaker, but only
        if the teams were evenly matched. Beating a weak team by 30 doesn't
        mean as much as beating a strong team by 30.
        """
        if not config.USE_MARGIN_OF_VICTORY:
            return 1.0  # Don't scale by margin of victory
        
        # Ties get no margin-of-victory bonus (but still get ELO change from the tie itself)
        if point_difference == 0:
            return 1.0
        
        margin = abs(point_difference)
        
        if config.MOV_SCALING_METHOD == "elo_standard":
            # The standard ELO formula for margin of victory
            # Bigger margins matter more, but less so when teams are mismatched
            elo_difference = abs(team_a_effective_elo - team_b_effective_elo)
            mov_factor = np.log(margin + 1) * (2.2 / ((elo_difference * 0.001 + 2.2)))
        elif config.MOV_SCALING_METHOD == "log":
            # Logarithmic scaling: diminishing returns on bigger margins
            mov_factor = np.log(margin + 1) / np.log(21)  # Normalized so 20 points ≈ factor of 1
        elif config.MOV_SCALING_METHOD == "sqrt":
            # Square root scaling: moderate returns on bigger margins
            mov_factor = np.sqrt(margin) / np.sqrt(20)  # Normalized so 20 points ≈ factor of 1
        else:
            mov_factor = 1.0
        
        # Cap the factor to prevent extreme values from dominating
        return min(mov_factor, 2.0)
    
    def apply_season_reset_at_start_of_new_season(new_season: int):
        """
        Apply regression to the mean at the start of a new season.
        
        This prevents ratings from drifting too far from the baseline over time.
        A lambda of 0.15 means we move 15% of the way back toward the starting
        ELO (1000) at the start of each season.
        """
        if not config.USE_SEASON_RESET:
            return
        
        regression_factor = config.SEASON_RESET_LAMBDA
        
        for team_name in current_ratings:
            rating_at_end_of_last_season = current_ratings[team_name]
            
            # Blend the end-of-season rating with the starting rating
            # Higher lambda = more regression toward the mean
            new_rating = ((1 - regression_factor) * rating_at_end_of_last_season + 
                         regression_factor * config.START_ELO)
            
            current_ratings[team_name] = new_rating
    
    # Step 5: Process each game chronologically
    # This is where the magic happens - we update ratings after each game
    game_results = []
    current_season_being_processed = None
    
    print("  Processing games chronologically...")
    for _, game_row in unique_games.iterrows():
        season = int(game_row["Season"])
        
        # Check if we're starting a new season
        if current_season_being_processed is not None and season != current_season_being_processed:
            # Save the ratings from the end of the previous season
            ratings_at_end_of_season[current_season_being_processed] = current_ratings.copy()
            # Apply the season reset (regression toward mean)
            apply_season_reset_at_start_of_new_season(season)
        
        current_season_being_processed = season
        
        # Get the two teams playing
        team_a_name = game_row["TeamA"]
        team_b_name = game_row["TeamB"]
        
        # Get their current ELO ratings
        team_a_rating = get_current_rating(team_a_name)
        team_b_rating = get_current_rating(team_b_name)
        
        # Apply home field advantage
        # Home teams get a boost to their effective rating
        team_a_is_home = game_row["A_Home"] and not game_row["B_Home"]
        team_b_is_home = game_row["B_Home"] and not game_row["A_Home"]
        
        team_a_home_advantage = config.HOME_FIELD_ADVANTAGE if team_a_is_home else 0.0
        team_b_home_advantage = config.HOME_FIELD_ADVANTAGE if team_b_is_home else 0.0
        
        team_a_effective_rating = team_a_rating + team_a_home_advantage
        team_b_effective_rating = team_b_rating + team_b_home_advantage
        
        # Calculate expected win probability for Team A
        # This is the core ELO formula: probability = 1 / (1 + 10^(rating_diff/400))
        rating_difference = team_b_effective_rating - team_a_effective_rating
        team_a_expected_win_probability = 1.0 / (1.0 + 10 ** (rating_difference / 400.0))
        team_b_expected_win_probability = 1.0 - team_a_expected_win_probability
        
        # Determine actual outcome
        # 1.0 = win, 0.5 = tie, 0.0 = loss
        game_outcome = game_row["Outcome"]
        if game_outcome == "W":
            team_a_actual_score = 1.0
            team_b_actual_score = 0.0
        elif game_outcome == "T":
            team_a_actual_score = 0.5
            team_b_actual_score = 0.5
        else:  # Loss
            team_a_actual_score = 0.0
            team_b_actual_score = 1.0
        
        # Calculate margin of victory factor
        point_difference = game_row["TeamPoints"] - game_row["OppPoints"]
        mov_scaling_factor = calculate_margin_of_victory_factor(
            point_difference, 
            team_a_effective_rating, 
            team_b_effective_rating
        )
        
        # Calculate rating changes
        # The change = K-factor × MOV-factor × (actual - expected)
        # If you win when you were expected to lose, you gain a lot
        # If you win when you were expected to win, you gain a little
        team_a_rating_change = (config.K_FACTOR * mov_scaling_factor * 
                                (team_a_actual_score - team_a_expected_win_probability))
        team_b_rating_change = (config.K_FACTOR * mov_scaling_factor * 
                                (team_b_actual_score - team_b_expected_win_probability))
        
        # Update the ratings
        team_a_new_rating = team_a_rating + team_a_rating_change
        team_b_new_rating = team_b_rating + team_b_rating_change
        
        current_ratings[team_a_name] = team_a_new_rating
        current_ratings[team_b_name] = team_b_new_rating
        
        # Track league-wide average if enabled
        # This helps us see if the league is getting stronger/weaker over time
        if config.TRACK_LEAGUE_MEAN:
            average_rating_across_all_teams = np.mean(list(current_ratings.values()))
            league_average_over_time.append({
                "Date": game_row["Date"],
                "Season": season,
                "LeagueMean": average_rating_across_all_teams,
                "GameID": int(game_row["GameID"])
            })
        
        # Store all the details for this game
        game_results.append({
            "Date": game_row["Date"], 
            "Season": season, 
            "GameID": int(game_row["GameID"]),
            "TeamA": team_a_name, 
            "TeamB": team_b_name,
            "A_Pre": team_a_rating, 
            "B_Pre": team_b_rating, 
            "A_Exp": team_a_expected_win_probability, 
            "B_Exp": team_b_expected_win_probability,
            "A_Act": team_a_actual_score, 
            "B_Act": team_b_actual_score, 
            "A_Delta": team_a_rating_change, 
            "B_Delta": team_b_rating_change,
            "A_Post": team_a_new_rating, 
            "B_Post": team_b_new_rating,
            "A_Home": game_row["A_Home"], 
            "B_Home": game_row["B_Home"],
            "A_Points": game_row["TeamPoints"], 
            "B_Points": game_row["OppPoints"],
            "PointDiff": point_difference,
            "MOV_Factor": mov_scaling_factor,
            "Outcome": game_outcome
        })
    
    # Save the final season's ratings
    if current_season_being_processed is not None:
        ratings_at_end_of_season[current_season_being_processed] = current_ratings.copy()
    
    # Convert results to DataFrame
    elo_games = pd.DataFrame(game_results)
    
    # Merge in league mean data if we tracked it
    if config.TRACK_LEAGUE_MEAN and league_average_over_time:
        league_mean_dataframe = pd.DataFrame(league_average_over_time)
        # Create a unique key to match games with their league mean
        league_mean_dataframe["DateGameID"] = (
            league_mean_dataframe["Date"].astype(str) + "_" + 
            league_mean_dataframe["GameID"].astype(str)
        )
        elo_games["DateGameID"] = (
            elo_games["Date"].astype(str) + "_" + 
            elo_games["GameID"].astype(str)
        )
        elo_games = elo_games.merge(
            league_mean_dataframe[["DateGameID", "LeagueMean"]],
            on="DateGameID",
            how="left"
        )
        elo_games = elo_games.drop(columns=["DateGameID"])
    
    print(f"  [OK] Calculated ELO ratings for {len(elo_games):,} games")
    if config.USE_MARGIN_OF_VICTORY:
        print(f"  [OK] Using margin of victory scaling: {config.MOV_SCALING_METHOD}")
    if config.USE_SEASON_RESET:
        print(f"  [OK] Using season reset with lambda={config.SEASON_RESET_LAMBDA}")
    
    return elo_games

def create_team_timeline(elo_games: pd.DataFrame) -> pd.DataFrame:
    """
    Create a timeline of ELO ratings for each team.
    
    This reorganizes the game-by-game data into a team-centric view,
    where each row represents one team's perspective of a game. This makes
    it much easier to analyze individual team performance over time.
    
    Args:
        elo_games: DataFrame with ELO game data (each game has TeamA and TeamB)
        
    Returns:
        DataFrame with team ELO timeline (each row is one team in one game)
    """
    print("Creating team ELO timeline...")
    
    # We'll build a list of timeline entries, one per team per game
    timeline_entries = []
    
    # Get all unique teams (they might appear as either TeamA or TeamB)
    all_teams_in_data = pd.concat([elo_games['TeamA'], elo_games['TeamB']]).unique()
    
    # For each team, find all their games and create timeline entries
    for team_name in all_teams_in_data:
        # Find all games where this team played (either as TeamA or TeamB)
        games_for_this_team = elo_games[
            (elo_games['TeamA'] == team_name) | (elo_games['TeamB'] == team_name)
        ].sort_values('Date')
        
        # Process each game from this team's perspective
        for _, game_row in games_for_this_team.iterrows():
            # Figure out if this team was TeamA or TeamB
            if game_row['TeamA'] == team_name:
                # This team was TeamA
                opponent_name = game_row['TeamB']
                team_was_home = game_row['A_Home']
                
                # Convert the numeric outcome (1.0, 0.5, 0.0) to a letter (W, T, L)
                actual_outcome_value = game_row['A_Act']
                if actual_outcome_value == 1.0:
                    outcome_letter = 'W'
                elif actual_outcome_value == 0.5:
                    outcome_letter = 'T'
                else:
                    outcome_letter = 'L'
                
                # Create timeline entry from TeamA's perspective
                timeline_entries.append({
                    'Date': game_row['Date'],
                    'Season': game_row['Season'],
                    'Team': team_name,
                    'Opponent': opponent_name,
                    'Elo_Pre': game_row['A_Pre'],
                    'Elo_Post': game_row['A_Post'],
                    'Elo_Change': game_row['A_Delta'],
                    'Expected': game_row['A_Exp'],
                    'Actual': game_row['A_Act'],
                    'Home': team_was_home,
                    'Points_For': game_row['A_Points'],
                    'Points_Against': game_row['B_Points'],
                    'Outcome': outcome_letter
                })
            else:
                # This team was TeamB
                opponent_name = game_row['TeamA']
                team_was_home = game_row['B_Home']
                
                # Convert the numeric outcome to a letter
                actual_outcome_value = game_row['B_Act']
                if actual_outcome_value == 1.0:
                    outcome_letter = 'W'
                elif actual_outcome_value == 0.5:
                    outcome_letter = 'T'
                else:
                    outcome_letter = 'L'
                
                # Create timeline entry from TeamB's perspective
                timeline_entries.append({
                    'Date': game_row['Date'],
                    'Season': game_row['Season'],
                    'Team': team_name,
                    'Opponent': opponent_name,
                    'Elo_Pre': game_row['B_Pre'],
                    'Elo_Post': game_row['B_Post'],
                    'Elo_Change': game_row['B_Delta'],
                    'Expected': game_row['B_Exp'],
                    'Actual': game_row['B_Act'],
                    'Home': team_was_home,
                    'Points_For': game_row['B_Points'],
                    'Points_Against': game_row['A_Points'],
                    'Outcome': outcome_letter
                })
    
    # Convert to DataFrame and sort by team and date
    timeline = pd.DataFrame(timeline_entries).sort_values(['Team', 'Date'])
    
    # Add rolling statistics (moving averages, etc.) to smooth out the data
    timeline = add_rolling_features(timeline, config.ROLLING_WINDOW_MONTHS)
    
    print(f"  [OK] Created timeline with {len(timeline):,} game records for {timeline['Team'].nunique()} teams")
    
    return timeline

def create_monthly_elo_data(timeline: pd.DataFrame) -> pd.DataFrame:
    """
    Create monthly aggregated ELO data for smooth plotting.
    
    This function replicates the team_monthly_elo structure from the original script
    to ensure smooth lines in the tapestry plot.
    
    Args:
        timeline: DataFrame with team ELO timeline
        
    Returns:
        DataFrame with monthly aggregated ELO data
    """
    print("Creating monthly ELO aggregation...")
    
    # Convert Date to YearMonth (monthly granularity)
    timeline_copy = timeline.copy()
    timeline_copy['YearMonth'] = timeline_copy['Date'].values.astype('datetime64[M]')
    
    # Get the last game of each month for each team (most recent ELO)
    valid = timeline_copy.dropna(subset=['Date']).copy()
    idx = valid.groupby(['Team', 'YearMonth'])['Date'].idxmax()
    
    team_monthly_last = (
        timeline_copy.loc[idx, ['Team', 'YearMonth', 'Elo_Post', 'Date']]
        .rename(columns={'Elo_Post': 'Elo_Last', 'Date': 'Date_Last'})
        .sort_values(['Team', 'YearMonth'])
        .reset_index(drop=True)
    )
    
    # Calculate monthly mean ELO for each team
    team_monthly_mean = (
        timeline_copy.groupby(['Team', 'YearMonth'], as_index=False)['Elo_Post']
        .mean()
        .rename(columns={'Elo_Post': 'Elo_Mean'})
    )
    
    # Combine last and mean ELO data
    team_monthly_elo = (
        team_monthly_last.merge(team_monthly_mean, on=['Team', 'YearMonth'], how='left')
        .sort_values(['Team', 'YearMonth'])
        .reset_index(drop=True)
    )
    
    # Rename Team to Franchise for consistency with original
    team_monthly_elo = team_monthly_elo.rename(columns={'Team': 'Franchise'})
    
    # Add rolling features for smooth plotting
    team_monthly_elo = add_rolling_features(team_monthly_elo, config.ROLLING_WINDOW_MONTHS)
    
    print(f"Created monthly ELO data with {len(team_monthly_elo)} records for {team_monthly_elo['Franchise'].nunique()} teams")
    
    return team_monthly_elo

def create_elo_games_data(timeline: pd.DataFrame) -> pd.DataFrame:
    """
    Create elo_games DataFrame with proper structure for calibration and upset analysis.
    This matches the original script's elo_games format.
    
    Args:
        timeline: DataFrame with game-by-game ELO data
        
    Returns:
        DataFrame with elo_games structure (A_Exp, A_Res, A_Pre, B_Pre, etc.)
    """
    print("Creating elo_games data structure...")
    
    # Group by date and find matching team pairs
    games = []
    processed_games = set()
    
    for date, date_games in timeline.groupby('Date'):
        # Get unique team pairs for this date
        team_pairs = set()
        for _, row in date_games.iterrows():
            team_pair = tuple(sorted([row['Team'], row['Opponent']]))
            team_pairs.add(team_pair)
        
        # Process each unique team pair
        for team_a, team_b in team_pairs:
            game_key = (date, team_a, team_b)
            if game_key in processed_games:
                continue
                
            # Get data for both teams
            team_a_data = date_games[date_games['Team'] == team_a]
            team_b_data = date_games[date_games['Team'] == team_b]
            
            if len(team_a_data) == 1 and len(team_b_data) == 1:
                team_a_row = team_a_data.iloc[0]
                team_b_row = team_b_data.iloc[0]
                
                # Create game record in elo_games format
                game_record = {
                    'Date': date,
                    'Season': team_a_row['Season'],
                    'GameID': len(games) + 1,
                    'TeamA': team_a,
                    'TeamB': team_b,
                    'A_Pre': team_a_row['Elo_Pre'],
                    'B_Pre': team_b_row['Elo_Pre'],
                    'A_Exp': team_a_row['Expected'],
                    'B_Exp': team_b_row['Expected'],
                    'A_Res': 1.0 if team_a_row['Outcome'] == 'W' else 0.5 if team_a_row['Outcome'] == 'T' else 0.0,
                    'B_Res': 1.0 if team_b_row['Outcome'] == 'W' else 0.5 if team_b_row['Outcome'] == 'T' else 0.0,
                    'A_Act': 1.0 if team_a_row['Outcome'] == 'W' else 0.5 if team_a_row['Outcome'] == 'T' else 0.0,
                    'B_Act': 1.0 if team_b_row['Outcome'] == 'W' else 0.5 if team_b_row['Outcome'] == 'T' else 0.0,
                    'A_Post': team_a_row['Elo_Post'],
                    'B_Post': team_b_row['Elo_Post'],
                    'A_Pts': team_a_row['Points_For'],
                    'B_Pts': team_b_row['Points_For'],
                    'PointDiff': team_a_row['Points_For'] - team_b_row['Points_For'],
                    'A_Home': team_a_row['Home'],
                    'B_Home': team_b_row['Home']
                }
                games.append(game_record)
                processed_games.add(game_key)
    
    elo_games = pd.DataFrame(games)
    print(f"Created elo_games with {len(elo_games)} games")
    return elo_games

# =============================================================================
# ADDITIONAL ANALYSIS FUNCTIONS
# =============================================================================

def calculate_team_summary_stats(timeline: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate summary statistics for each team.
    
    Args:
        timeline: DataFrame with team ELO timeline
        
    Returns:
        DataFrame with team summary statistics
    """
    print("Calculating team summary statistics...")
    
    summary_stats = []
    
    for team in timeline['Team'].unique():
        team_data = timeline[timeline['Team'] == team]
        
        # Basic stats
        current_elo = team_data['Elo_Post'].iloc[-1]
        peak_elo = team_data['Elo_Post'].max()
        low_elo = team_data['Elo_Post'].min()
        avg_elo = team_data['Elo_Post'].mean()
        
        # Win/loss stats
        wins = (team_data['Outcome'] == 'W').sum()
        losses = (team_data['Outcome'] == 'L').sum()
        ties = (team_data['Outcome'] == 'T').sum()
        total_games = len(team_data)
        win_pct = wins / total_games if total_games > 0 else 0
        
        # ELO change stats
        elo_change = current_elo - config.START_ELO
        max_gain = team_data['Elo_Change'].max()
        max_loss = team_data['Elo_Change'].min()
        
        # Season stats
        seasons = team_data['Season'].nunique()
        first_season = team_data['Season'].min()
        last_season = team_data['Season'].max()
        
        summary_stats.append({
            'Team': team,
            'Current_ELO': current_elo,
            'Peak_ELO': peak_elo,
            'Low_ELO': low_elo,
            'Avg_ELO': avg_elo,
            'ELO_Change_Total': elo_change,
            'Wins': wins,
            'Losses': losses,
            'Ties': ties,
            'Total_Games': total_games,
            'Win_Pct': win_pct,
            'Max_ELO_Gain': max_gain,
            'Max_ELO_Loss': max_loss,
            'Seasons': seasons,
            'First_Season': first_season,
            'Last_Season': last_season
        })
    
    summary_df = pd.DataFrame(summary_stats).sort_values('Current_ELO', ascending=False)
    print(f"Calculated summary stats for {len(summary_df)} teams")
    
    # Print top teams by average ELO for verification
    top_avg_elo = summary_df.nlargest(5, 'Avg_ELO')[['Team', 'Avg_ELO', 'Total_Games', 'Seasons']]
    print(f"\n  Top 5 teams by Average ELO:")
    for _, row in top_avg_elo.iterrows():
        print(f"    {row['Team']}: {row['Avg_ELO']:.2f} (over {row['Total_Games']:.0f} games, {row['Seasons']:.0f} seasons)")
    
    return summary_df

def find_biggest_swings(timeline: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Find the biggest ELO swings (gains and losses) in NFL history.
    
    Args:
        timeline: DataFrame with team ELO timeline
        top_n: Number of top swings to return
        
    Returns:
        DataFrame with biggest swings
    """
    print(f"Finding top {top_n} biggest ELO swings...")
    
    # Get biggest gains and losses
    biggest_gains = timeline.nlargest(top_n, 'Elo_Change')[['Date', 'Team', 'Opponent', 'Elo_Change', 'Outcome', 'Points_For', 'Points_Against']]
    biggest_losses = timeline.nsmallest(top_n, 'Elo_Change')[['Date', 'Team', 'Opponent', 'Elo_Change', 'Outcome', 'Points_For', 'Points_Against']]
    
    biggest_gains['Swing_Type'] = 'Gain'
    biggest_losses['Swing_Type'] = 'Loss'
    
    # Combine and sort by absolute change
    all_swings = pd.concat([biggest_gains, biggest_losses], ignore_index=True)
    all_swings['Abs_Change'] = all_swings['Elo_Change'].abs()
    all_swings = all_swings.sort_values('Abs_Change', ascending=False).head(top_n)
    
    print(f"Found {len(all_swings)} biggest swings")
    
    return all_swings

def create_season_team_summary(timeline: pd.DataFrame) -> pd.DataFrame:
    """
    Create season team summary with start/end ELO ratings and rankings.
    
    Args:
        timeline: DataFrame with team ELO timeline
        
    Returns:
        DataFrame with season team summaries
    """
    print("Creating season team summary...")
    
    summary_data = []
    
    for season in sorted(timeline['Season'].unique()):
        season_data = timeline[timeline['Season'] == season]
        
        for team in season_data['Team'].unique():
            team_season = season_data[season_data['Team'] == team].sort_values('Date')
            
            if not team_season.empty:
                summary_data.append({
                    'Season': season,
                    'Team': team,
                    'Elo_Pre': team_season['Elo_Pre'].iloc[0],  # Start of season
                    'Elo_Post': team_season['Elo_Post'].iloc[-1],  # End of season
                    'Games': len(team_season),
                    'Wins': (team_season['Outcome'] == 'W').sum(),
                    'Losses': (team_season['Outcome'] == 'L').sum(),
                    'Ties': (team_season['Outcome'] == 'T').sum()
                })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Add rankings
    summary_df['Start_Rank'] = summary_df.groupby('Season')['Elo_Pre'].rank(ascending=False, method='dense')
    summary_df['End_Rank'] = summary_df.groupby('Season')['Elo_Post'].rank(ascending=False, method='dense')
    
    print(f"Created season team summary with {len(summary_df)} records")
    
    return summary_df

def get_team_conference(team: str, season: int) -> str:
    """
    Get team's conference for a given season, accounting for 1976 expansion and 2002 realignment.
    
    Historical quirks:
    - SEA: NFC (1976), AFC (1977-2001), NFC (2002+)
    - TAM: AFC (1976), NFC (1977+)
    
    Args:
        team: Team code
        season: Season year
        
    Returns:
        Conference ('AFC' or 'NFC')
    """
    # Seahawks special: NFC (1976), AFC (1977-2001), NFC (2002+)
    if team == 'SEA':
        if season == 1976:
            return 'NFC'
        elif season < 2002:
            return 'AFC'
        else:
            return 'NFC'
    
    # Buccaneers special: AFC (1976), NFC (1977+)
    if team == 'TAM':
        return 'AFC' if season == 1976 else 'NFC'
    
    # All other teams - use post-2002 mapping (most teams didn't change conference)
    conference_map = {
        'ARI': 'NFC', 'ATL': 'NFC', 'BAL': 'AFC', 'BUF': 'AFC', 'CAR': 'NFC', 'CHI': 'NFC',
        'CIN': 'AFC', 'CLE': 'AFC', 'DAL': 'NFC', 'DEN': 'AFC', 'DET': 'NFC', 'GNB': 'NFC',
        'HOU': 'AFC', 'IND': 'AFC', 'JAX': 'AFC', 'KAN': 'AFC', 'LVR': 'AFC', 'LAC': 'AFC',
        'LAR': 'NFC', 'MIA': 'AFC', 'MIN': 'NFC', 'NWE': 'AFC', 'NOR': 'NFC', 'NYG': 'NFC',
        'NYJ': 'AFC', 'PHI': 'NFC', 'PIT': 'AFC', 'SFO': 'NFC',
        'TEN': 'AFC', 'WAS': 'NFC'
    }
    return conference_map.get(team, 'Unknown')  # Return 'Unknown' instead of defaulting to catch bugs

def get_team_division(team: str, season: int) -> str:
    """
    Get team's division for a given season, accounting for 1976 expansion and 2002 realignment.
    
    Historical quirks:
    - 1976: SEA in NFC West, TAM in AFC West
    - 1977-2001: SEA in AFC West, TAM in NFC Central
    - 2002+: Current 8-division structure
    
    Args:
        team: Team code
        season: Season year
        
    Returns:
        Division name (e.g., 'AFC East', 'NFC West')
    """
    if season < 2002:
        # 1976 expansion quirks
        if season == 1976:
            if team == 'SEA':
                return 'NFC West'
            if team == 'TAM':
                return 'AFC West'
        
        # Pre-2002 structure (6 divisions) - 1977-2001 era
        pre_2002_divisions = {
            # AFC East (5 teams)
            'BUF': 'AFC East', 'IND': 'AFC East', 'MIA': 'AFC East', 
            'NWE': 'AFC East', 'NYJ': 'AFC East',
            # AFC Central (6 teams)
            'BAL': 'AFC Central', 'CIN': 'AFC Central', 'CLE': 'AFC Central',
            'JAX': 'AFC Central', 'PIT': 'AFC Central', 'TEN': 'AFC Central',
            # AFC West (5 teams)
            'DEN': 'AFC West', 'KAN': 'AFC West', 'LVR': 'AFC West',
            'LAC': 'AFC West', 'SEA': 'AFC West',  # SEA was AFC 1977-2001
            # NFC East (5 teams)
            'ARI': 'NFC East', 'DAL': 'NFC East', 'NYG': 'NFC East',
            'PHI': 'NFC East', 'WAS': 'NFC East',
            # NFC Central (5 teams)
            'CHI': 'NFC Central', 'DET': 'NFC Central', 'GNB': 'NFC Central',
            'MIN': 'NFC Central', 'TAM': 'NFC Central',  # TAM was NFC Central 1977-2001
            # NFC West (5 teams)
            'ATL': 'NFC West', 'CAR': 'NFC West', 'NOR': 'NFC West',
            'SFO': 'NFC West', 'STL': 'NFC West', 'LAR': 'NFC West'  # STL/LAR both map to same
        }
        return pre_2002_divisions.get(team, 'Unknown')
    else:
        # Post-2002 structure (8 divisions)
        post_2002_divisions = {
            'ARI': 'NFC West', 'ATL': 'NFC South', 'BAL': 'AFC North', 'BUF': 'AFC East',
            'CAR': 'NFC South', 'CHI': 'NFC North', 'CIN': 'AFC North', 'CLE': 'AFC North',
            'DAL': 'NFC East', 'DEN': 'AFC West', 'DET': 'NFC North', 'GNB': 'NFC North',
            'HOU': 'AFC South', 'IND': 'AFC South', 'JAX': 'AFC South', 'KAN': 'AFC West',
            'LVR': 'AFC West', 'LAC': 'AFC West', 'LAR': 'NFC West', 'MIA': 'AFC East',
            'MIN': 'NFC North', 'NWE': 'AFC East', 'NOR': 'NFC South', 'NYG': 'NFC East',
            'NYJ': 'AFC East', 'PHI': 'NFC East', 'PIT': 'AFC North', 'SFO': 'NFC West',
            'SEA': 'NFC West', 'TAM': 'NFC South', 'TEN': 'AFC South', 'WAS': 'NFC East',
            'STL': 'NFC West'  # STL codes get canonicalized to LAR, but handle both
        }
        return post_2002_divisions.get(team, 'Unknown')

def create_season_conference_summary(season_team_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Create season conference summary with average ELO by conference.
    Accounts for 2002 realignment (SEA switched from AFC to NFC).
    
    Args:
        season_team_summary: DataFrame with season team summaries
        
    Returns:
        DataFrame with conference summaries
    """
    print("Creating season conference summary...")
    
    summary_df = season_team_summary.copy()
    # Apply season-aware conference mapping
    summary_df['Conference'] = summary_df.apply(
        lambda row: get_team_conference(row['Team'], int(row['Season'])), axis=1
    )
    
    # Group by season and conference
    conference_summary = summary_df.groupby(['Season', 'Conference']).agg({
        'Elo_Pre': 'mean',
        'Elo_Post': 'mean',
        'Team': 'count'
    }).reset_index()
    
    conference_summary.columns = ['Season', 'Conference', 'EloMean', 'EloEnd', 'TeamCount']
    
    print(f"Created conference summary with {len(conference_summary)} records")
    
    return conference_summary

def build_luck_index(timeline: pd.DataFrame, elo_games: pd.DataFrame = None) -> pd.DataFrame:
    """
    Build Luck Index table with close-game and Pythagorean variants.
    
    Definitions:
        Luck_Index(team, season) = ActualWins − ExpectedWins (ELO-based)
        Close_Luck_Index = ActualWins in close games − ExpectedWins in close games
        Pythag_Luck = ActualWins − Pythagorean ExpectedWins
    
    Close game definition:
        - Point differential ≤ CLOSE_GAME_POINT_DIFF (default: 7)
        - OR predicted win probability in [0.45, 0.55]
    
    Interpretation:
        > 0 : team won more than model expected (good fortune/execution in key moments).
        < 0 : team underperformed expectation (bad luck/close-game woes).
    
    Returns:
        DataFrame with [Season, Team, Games, ActualWins, ExpWins, Luck_Index, Luck_per_Game,
                        Close_ActualWins, Close_ExpWins, Close_Luck_Index, Close_Luck_per_Game,
                        Pythag_ExpWins, Pythag_Luck].
    """
    print("Building luck index...")
    
    # Calculate expected wins from timeline data
    luck_data = []
    
    for season in sorted(timeline['Season'].unique()):
        season_data = timeline[timeline['Season'] == season]
        
        for team in season_data['Team'].unique():
            team_season = season_data[season_data['Team'] == team].sort_values('Date')
            
            if team_season.empty:
                continue
            
            # Basic luck metrics
            actual_wins = (team_season['Outcome'] == 'W').sum()
            expected_wins = team_season['Expected'].sum()
            games = len(team_season)
            
            luck_index = actual_wins - expected_wins
            luck_per_game = luck_index / games if games > 0 else 0
            
            # Close game luck
            close_point_diff = abs(team_season['Points_For'] - team_season['Points_Against']) <= config.CLOSE_GAME_POINT_DIFF
            close_prob_range = (team_season['Expected'] >= config.CLOSE_GAME_PROB_RANGE[0]) & \
                              (team_season['Expected'] <= config.CLOSE_GAME_PROB_RANGE[1])
            is_close_game = close_point_diff | close_prob_range
            
            close_games = team_season[is_close_game]
            close_actual_wins = (close_games['Outcome'] == 'W').sum() if not close_games.empty else 0
            close_expected_wins = close_games['Expected'].sum() if not close_games.empty else 0
            close_games_count = len(close_games)
            close_luck_index = close_actual_wins - close_expected_wins
            close_luck_per_game = close_luck_index / close_games_count if close_games_count > 0 else 0
            
            # Pythagorean expectation
            points_for = team_season['Points_For'].sum()
            points_against = team_season['Points_Against'].sum()
            
            if points_for > 0 and points_against > 0:
                pyth_win_pct = (points_for ** config.PYTHAG_EXPONENT) / \
                              (points_for ** config.PYTHAG_EXPONENT + points_against ** config.PYTHAG_EXPONENT)
                pyth_exp_wins = pyth_win_pct * games
                pyth_luck = actual_wins - pyth_exp_wins
            else:
                pyth_exp_wins = 0
                pyth_luck = 0
            
            luck_data.append({
                'Season': season,
                'Team': team,
                'Games': games,
                'ActualWins': actual_wins,
                'ExpWins': expected_wins,
                'Luck_Index': luck_index,
                'Luck_per_Game': luck_per_game,
                'Close_ActualWins': close_actual_wins,
                'Close_ExpWins': close_expected_wins,
                'Close_Games': close_games_count,
                'Close_Luck_Index': close_luck_index,
                'Close_Luck_per_Game': close_luck_per_game,
                'Points_For': points_for,
                'Points_Against': points_against,
                'Pyth_ExpWins': pyth_exp_wins,
                'Pyth_Luck': pyth_luck
            })
    
    luck_df = pd.DataFrame(luck_data)
    luck_df = luck_df.sort_values(['Season', 'Luck_Index'], ascending=[True, False]).reset_index(drop=True)
    
    print(f"Built luck index with {len(luck_df)} records")
    print(f"  Average close games per team-season: {luck_df['Close_Games'].mean():.1f}")
    
    # Print luck index statistics for documentation
    luck_values = luck_df['Luck_Index']
    print(f"\n  Luck Index Statistics:")
    print(f"    Min: {luck_values.min():.2f}")
    print(f"    5th percentile: {luck_values.quantile(0.05):.2f}")
    print(f"    25th percentile: {luck_values.quantile(0.25):.2f}")
    print(f"    Median: {luck_values.median():.2f}")
    print(f"    75th percentile: {luck_values.quantile(0.75):.2f}")
    print(f"    95th percentile: {luck_values.quantile(0.95):.2f}")
    print(f"    Max: {luck_values.max():.2f}")
    print(f"    Std Dev: {luck_values.std():.2f}")
    
    return luck_df

def calculate_average_luck_by_team(luck_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average luck scores for each team across all seasons.
    
    Args:
        luck_df: DataFrame with luck index data (from build_luck_index)
    
    Returns:
        DataFrame with [Team, Seasons, Avg_Luck_Index, Avg_Luck_per_Game, 
                        Avg_Close_Luck_Index, Avg_Close_Luck_per_Game, Avg_Pyth_Luck,
                        Total_Games, Total_ActualWins, Total_ExpWins]
    """
    print("Calculating average luck scores by team...")
    
    # Group by team and calculate averages
    team_avg_luck = luck_df.groupby('Team').agg({
        'Season': 'count',  # Number of seasons
        'Luck_Index': 'mean',
        'Luck_per_Game': 'mean',
        'Close_Luck_Index': 'mean',
        'Close_Luck_per_Game': 'mean',
        'Pyth_Luck': 'mean',
        'Games': 'sum',  # Total games across all seasons
        'ActualWins': 'sum',  # Total actual wins
        'ExpWins': 'sum'  # Total expected wins
    }).reset_index()
    
    # Rename columns for clarity
    team_avg_luck.columns = [
        'Team', 'Seasons', 'Avg_Luck_Index', 'Avg_Luck_per_Game',
        'Avg_Close_Luck_Index', 'Avg_Close_Luck_per_Game', 'Avg_Pyth_Luck',
        'Total_Games', 'Total_ActualWins', 'Total_ExpWins'
    ]
    
    # Calculate overall luck index (total actual - total expected)
    team_avg_luck['Overall_Luck_Index'] = team_avg_luck['Total_ActualWins'] - team_avg_luck['Total_ExpWins']
    team_avg_luck['Overall_Luck_per_Game'] = team_avg_luck['Overall_Luck_Index'] / team_avg_luck['Total_Games']
    
    # Sort by average luck index (most lucky first)
    team_avg_luck = team_avg_luck.sort_values('Avg_Luck_Index', ascending=False).reset_index(drop=True)
    
    print(f"Calculated average luck for {len(team_avg_luck)} teams")
    
    return team_avg_luck

def _safe_logloss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    """Numerically-stable binary log loss for outcomes y ∈ {0, 0.5, 1}, with ties treated as 0.5."""
    p = np.clip(p.astype(float), eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

def _brier(y: np.ndarray, p: np.ndarray) -> float:
    """Brier score (mean squared error) between predicted win prob and actual outcome."""
    return float(np.mean((y.astype(float) - p.astype(float)) ** 2))

def build_calibration_bins(
    elo_games: pd.DataFrame,
    bins: int = 10,
    min_bin_n: int = 50,
    groupby_cols: List[str] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create reliability-curve bins and season performance table.
    
    Supports segmented calibration by era, ELO difference, home/away, etc.

    Reliability Curve:
        - Bin games by predicted Pwin (quantiles).
        - Compare mean predicted Pwin to mean actual outcome per bin.
        - Perfect calibration lies on y=x.
        - Can be segmented by groupby_cols (e.g., ["Era10"], ["HomeFlag"], ["EloDiffBin"])

    Season Performance:
        - Brier and LogLoss by season to track forecast quality over time.

    Args:
        elo_games: DataFrame with ELO game data
        bins: Number of quantile bins
        min_bin_n: Minimum games per bin
        groupby_cols: Optional list of columns to segment calibration by
                     (e.g., ["Era10"], ["A_Home"], ["EloDiffBin"])

    Returns:
        curve: DataFrame with per-bin Pwin_mean, Outcome_mean, N (and groupby columns if specified).
        perf:  DataFrame with per-season Brier and LogLoss.
    """
    print("Building calibration bins...")
    g = elo_games.dropna(subset=["A_Exp","A_Act","Season"]).copy()
    g = g[g["A_Act"].isin([0.0, 0.5, 1.0])]  # ties counted as 0.5 in calibration
    
    # Add segmentation columns if needed
    if groupby_cols is None:
        groupby_cols = []
    
    # Add era segmentation if requested
    if "Era10" in groupby_cols and "Era10" not in g.columns:
        g["Era10"] = (g["Season"] // 10) * 10  # 1970s, 1980s, etc.
    
    # Add ELO difference bin if requested
    if "EloDiffBin" in groupby_cols and "EloDiffBin" not in g.columns:
        g["EloDiff"] = abs(g["A_Pre"] - g["B_Pre"])
        g["EloDiffBin"] = pd.cut(g["EloDiff"], bins=[0, 50, 100, 150, 200, 500], 
                                 labels=["0-50", "50-100", "100-150", "150-200", "200+"],
                                 include_lowest=True)
    
    # Add home flag if requested
    if "HomeFlag" in groupby_cols and "HomeFlag" not in g.columns:
        g["HomeFlag"] = g["A_Home"].map({1: "Home", 0: "Away"})
    
    # Create quantile bins
    if groupby_cols:
        # Bin within each group
        g["Bin"] = g.groupby(groupby_cols)["A_Exp"].transform(
            lambda x: pd.qcut(x, q=bins, duplicates="drop")
        )
    else:
        g["Bin"] = pd.qcut(g["A_Exp"], q=bins, duplicates="drop")
    
    # Group by bin and any segmentation columns
    group_cols = groupby_cols + ["Bin"]
    curve = (
        g.groupby(group_cols, observed=True)
         .agg(
             Pwin_mean=("A_Exp","mean"),
             Outcome_mean=("A_Act","mean"),
             N=("A_Act","size")
         )
         .reset_index()
    )
    curve = curve[curve["N"] >= min_bin_n].sort_values(groupby_cols + ["Pwin_mean"]).reset_index(drop=True)

    perf = (
        g.groupby("Season", as_index=False)
         .apply(lambda x: pd.Series({
             "Brier": _brier(x["A_Act"].to_numpy(), x["A_Exp"].to_numpy()),
             "LogLoss": _safe_logloss(x["A_Act"].to_numpy(), x["A_Exp"].to_numpy())
         }))
         .reset_index(drop=True)
         .sort_values("Season")
    )
    
    seg_info = f" segmented by {groupby_cols}" if groupby_cols else ""
    print(f"Built calibration with {len(curve)} bins{seg_info} and {len(perf)} seasons")
    return curve, perf

def calculate_brier_decomposition(elo_games: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Brier score decomposition: Brier = Uncertainty - Resolution + Reliability.
    
    Murphy decomposition:
    - Uncertainty: p̄(1 - p̄) where p̄ is mean outcome (climatology)
    - Resolution: Σ (n_k/N) * (p̄_k - p̄)² where p̄_k is mean outcome in bin k
    - Reliability: Σ (n_k/N) * (p̄_k - f_k)² where f_k is mean forecast in bin k
    
    Args:
        elo_games: DataFrame with ELO game data
        
    Returns:
        DataFrame with [Season, Brier, Uncertainty, Resolution, Reliability]
    """
    print("Calculating Brier decomposition...")
    
    g = elo_games.dropna(subset=["A_Exp", "A_Act", "Season"]).copy()
    g = g[g["A_Act"].isin([0.0, 0.5, 1.0])]  # ties counted as 0.5
    
    decomposition_data = []
    
    for season in sorted(g['Season'].unique()):
        season_games = g[g['Season'] == season]
        
        if season_games.empty:
            continue
        
        y = season_games['A_Act'].to_numpy()
        p = season_games['A_Exp'].to_numpy()
        n = len(season_games)
        
        # Brier score
        brier = _brier(y, p)
        
        # Uncertainty: p̄(1 - p̄) where p̄ is mean outcome
        p_bar = y.mean()
        uncertainty = p_bar * (1 - p_bar)
        
        # Bin forecasts for resolution and reliability calculation
        # Use quantile bins (10 bins)
        try:
            bins = pd.qcut(p, q=10, duplicates='drop')
            bin_stats = season_games.groupby(bins, observed=True).agg({
                'A_Act': ['mean', 'count'],
                'A_Exp': 'mean'
            }).reset_index()
            
            # Resolution: Σ (n_k/N) * (p̄_k - p̄)²
            resolution = 0.0
            # Reliability: Σ (n_k/N) * (p̄_k - f_k)²
            reliability = 0.0
            
            for _, row in bin_stats.iterrows():
                n_k = row[('A_Act', 'count')]
                p_k = row[('A_Act', 'mean')]  # Mean outcome in bin
                f_k = row[('A_Exp', 'mean')]  # Mean forecast in bin
                
                weight = n_k / n
                resolution += weight * (p_k - p_bar) ** 2
                reliability += weight * (p_k - f_k) ** 2
            
        except (ValueError, KeyError):
            # Fallback if binning fails (e.g., all probabilities same)
            resolution = 0.0
            reliability = brier - uncertainty + resolution  # Solve for reliability
        
        decomposition_data.append({
            'Season': season,
            'Brier': brier,
            'Uncertainty': uncertainty,
            'Resolution': resolution,
            'Reliability': reliability,
            'Games': n
        })
    
    decomp_df = pd.DataFrame(decomposition_data)
    
    print(f"Calculated Brier decomposition for {len(decomp_df)} seasons")
    print(f"  Average Brier: {decomp_df['Brier'].mean():.4f}")
    print(f"  Average Uncertainty: {decomp_df['Uncertainty'].mean():.4f}")
    print(f"  Average Resolution: {decomp_df['Resolution'].mean():.4f}")
    print(f"  Average Reliability: {decomp_df['Reliability'].mean():.4f}")
    
    return decomp_df

def calculate_elo_scores_only(games_preprocessed: pd.DataFrame,
                              k_factor: float,
                              home_advantage: float,
                              season_reset_lambda: float,
                              scale_constant: float) -> Tuple[float, float]:
    """
    Calculate only Brier score and LogLoss for sensitivity analysis (optimized).
    
    This version skips building the full DataFrame and only tracks what's needed.
    
    Args:
        games_preprocessed: Pre-processed games DataFrame (already deduplicated)
        k_factor: K-factor for rating volatility
        home_advantage: Home field advantage in ELO points
        season_reset_lambda: Season reset parameter
        scale_constant: Scale constant in ELO formula
        
    Returns:
        Tuple of (brier_score, logloss)
    """
    # Initialize ratings
    ratings: Dict[str, float] = {}
    START_ELO = config.START_ELO
    
    def get_elo(team: str) -> float:
        return ratings.get(team, START_ELO)
    
    # Pre-calculate constants
    USE_MOV = config.USE_MARGIN_OF_VICTORY
    MOV_METHOD = config.MOV_SCALING_METHOD
    USE_RESET = config.USE_SEASON_RESET
    
    # Track predictions and outcomes for scoring
    predictions = []
    outcomes = []
    
    current_season = None
    
    # Use itertuples() instead of iterrows() - MUCH faster
    for row in games_preprocessed.itertuples():
        season = row.Season
        
        # Apply season reset if needed
        if current_season is not None and season != current_season and USE_RESET:
            lambda_val = season_reset_lambda
            for team in ratings:
                elo_end = ratings[team]
                elo_new = (1 - lambda_val) * elo_end + lambda_val * START_ELO
                ratings[team] = elo_new
        
        current_season = season
        
        a, b = row.TeamA, row.TeamB
        Ra, Rb = get_elo(a), get_elo(b)
        
        # Home field advantage
        hfa_a = home_advantage if row.A_Home and not row.B_Home else 0.0
        hfa_b = home_advantage if row.B_Home and not row.A_Home else 0.0
        Ra_eff, Rb_eff = Ra + hfa_a, Rb + hfa_b
        
        # Expected scores
        Ea = 1.0 / (1.0 + 10 ** ((Rb_eff - Ra_eff) / scale_constant))
        Eb = 1.0 - Ea
        
        # Actual scores
        Sa = 1.0 if row.Outcome == "W" else 0.5 if row.Outcome == "T" else 0.0
        Sb = 1.0 - Sa if row.Outcome != "T" else 0.5
        
        # Calculate margin of victory factor
        point_diff = row.TeamPoints - row.OppPoints
        if USE_MOV and point_diff != 0:
            margin = abs(point_diff)
            if MOV_METHOD == "elo_standard":
                mov_factor = np.log(margin + 1) * (2.2 / ((abs(Ra_eff - Rb_eff) * 0.001 + 2.2)))
            else:
                mov_factor = 1.0
            mov_factor = min(mov_factor, 2.0)
        else:
            mov_factor = 1.0
        
        # Rating changes
        delta_a = k_factor * mov_factor * (Sa - Ea)
        delta_b = k_factor * mov_factor * (Sb - Eb)
        
        Ra_new, Rb_new = Ra + delta_a, Rb + delta_b
        ratings[a], ratings[b] = Ra_new, Rb_new
        
        # Store for scoring (only need A-side)
        predictions.append(Ea)
        outcomes.append(Sa)
    
    # Calculate scores
    y = np.array(outcomes, dtype=float)
    p = np.array(predictions, dtype=float)
    
    brier = _brier(y, p)
    logloss = _safe_logloss(y, p)
    
    return brier, logloss

def calculate_elo_with_params(df: pd.DataFrame, 
                           k_factor: float = None,
                           home_advantage: float = None,
                           season_reset_lambda: float = None,
                           scale_constant: float = None) -> pd.DataFrame:
    """
    Calculate ELO ratings with specific parameter values (for sensitivity analysis).
    
    This function temporarily overrides config parameters, runs the ELO calculation,
    then restores the original values.
    
    Args:
        df: DataFrame with game data
        k_factor: K-factor for rating volatility (default: config.K_FACTOR)
        home_advantage: Home field advantage in ELO points (default: config.HOME_FIELD_ADVANTAGE)
        season_reset_lambda: Season reset parameter (default: config.SEASON_RESET_LAMBDA)
        scale_constant: Scale constant in ELO formula (default: 400)
        
    Returns:
        DataFrame with ELO calculations
    """
    # Store original values
    original_k = config.K_FACTOR
    original_hfa = config.HOME_FIELD_ADVANTAGE
    original_lambda = config.SEASON_RESET_LAMBDA
    
    # Set new values
    if k_factor is not None:
        config.K_FACTOR = k_factor
    if home_advantage is not None:
        config.HOME_FIELD_ADVANTAGE = home_advantage
    if season_reset_lambda is not None:
        config.SEASON_RESET_LAMBDA = season_reset_lambda
    
    # Use provided scale constant or default
    scale = scale_constant if scale_constant is not None else 400.0
    
    # Prepare games data
    games_prep = (
        df.rename(columns={"Team_FR": "TeamA", "Opp_FR": "TeamB"})
        .sort_values(["Date", "GameID"])[
            ["Date", "GameID", "Season", "TeamA", "TeamB", "A_Home", "B_Home",
             "TeamPoints", "OppPoints", "Outcome"]
        ]
        .reset_index(drop=True)
    )
    
    # Deduplicate
    games_prep['GameKey'] = (
        games_prep['Date'].astype(str) + '_' + 
        games_prep[['TeamA', 'TeamB']].apply(lambda x: '_'.join(sorted([x['TeamA'], x['TeamB']])), axis=1)
    )
    games = games_prep.drop_duplicates(subset=['GameKey'], keep='first').drop(columns=['GameKey'])
    
    # Initialize ratings
    ratings: Dict[str, float] = {}
    season_end_ratings: Dict[int, Dict[str, float]] = {}
    
    def get_elo(team: str) -> float:
        return ratings.get(team, config.START_ELO)
    
    def calculate_mov_factor(point_diff: float, ra_eff: float, rb_eff: float) -> float:
        if not config.USE_MARGIN_OF_VICTORY:
            return 1.0
        if point_diff == 0:
            return 1.0
        margin = abs(point_diff)
        if config.MOV_SCALING_METHOD == "elo_standard":
            mov_factor = np.log(margin + 1) * (2.2 / ((abs(ra_eff - rb_eff) * 0.001 + 2.2)))
        else:
            mov_factor = 1.0
        return min(mov_factor, 2.0)
    
    def apply_season_reset(season: int):
        if not config.USE_SEASON_RESET:
            return
        lambda_val = config.SEASON_RESET_LAMBDA
        for team in ratings:
            elo_end = ratings[team]
            elo_new = (1 - lambda_val) * elo_end + lambda_val * config.START_ELO
            ratings[team] = elo_new
    
    # Process each game
    rows = []
    current_season = None
    
    try:
        for _, g in games.iterrows():
            season = int(g["Season"])
            
            if current_season is not None and season != current_season:
                season_end_ratings[current_season] = ratings.copy()
                apply_season_reset(season)
            
            current_season = season
            
            a, b = g["TeamA"], g["TeamB"]
            Ra, Rb = get_elo(a), get_elo(b)
            
            # Home field advantage
            hfa_a = config.HOME_FIELD_ADVANTAGE if g["A_Home"] and not g["B_Home"] else 0.0
            hfa_b = config.HOME_FIELD_ADVANTAGE if g["B_Home"] and not g["A_Home"] else 0.0
            Ra_eff, Rb_eff = Ra + hfa_a, Rb + hfa_b
            
            # Expected scores (using scale constant)
            Ea = 1.0 / (1.0 + 10 ** ((Rb_eff - Ra_eff) / scale))
            Eb = 1.0 - Ea
            
            # Actual scores
            Sa = 1.0 if g["Outcome"] == "W" else 0.5 if g["Outcome"] == "T" else 0.0
            Sb = 1.0 - Sa if g["Outcome"] != "T" else 0.5
            
            # Calculate margin of victory factor
            point_diff = g["TeamPoints"] - g["OppPoints"]
            mov_factor = calculate_mov_factor(point_diff, Ra_eff, Rb_eff)
            
            # Rating changes
            delta_a = config.K_FACTOR * mov_factor * (Sa - Ea)
            delta_b = config.K_FACTOR * mov_factor * (Sb - Eb)
            
            Ra_new, Rb_new = Ra + delta_a, Rb + delta_b
            ratings[a], ratings[b] = Ra_new, Rb_new
            
            rows.append({
                "Date": g["Date"], "Season": season, "GameID": int(g["GameID"]),
                "TeamA": a, "TeamB": b,
                "A_Pre": Ra, "B_Pre": Rb, "A_Exp": Ea, "B_Exp": Eb,
                "A_Act": Sa, "B_Act": Sb, "A_Delta": delta_a, "B_Delta": delta_b,
                "A_Post": Ra_new, "B_Post": Rb_new,
                "A_Home": g["A_Home"], "B_Home": g["B_Home"],
                "A_Points": g["TeamPoints"], "B_Points": g["OppPoints"],
                "PointDiff": point_diff,
                "MOV_Factor": mov_factor,
                "Outcome": g["Outcome"]
            })
        
        if current_season is not None:
            season_end_ratings[current_season] = ratings.copy()
        
        elo_games = pd.DataFrame(rows)
        
    finally:
        # Restore original values
        config.K_FACTOR = original_k
        config.HOME_FIELD_ADVANTAGE = original_hfa
        config.SEASON_RESET_LAMBDA = original_lambda
    
    return elo_games

def run_sensitivity_analysis(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Run sensitivity analysis by testing ALL combinations of parameter values.
    
    Tests full grid search of:
    - K-factor: 15, 20, 25, 30
    - Home field advantage: 40, 50, 55, 60, 70
    - Season reset lambda: 0.0, 0.10, 0.15, 0.20
    - Scale constant: 350, 400, 450
    
    Total combinations: 4 * 5 * 4 * 3 = 240
    
    OPTIMIZED: Pre-processes data once and uses fast itertuples() instead of iterrows()
    
    Args:
        raw_data: Raw game data DataFrame
        
    Returns:
        DataFrame with columns: K_Factor, Home_Advantage, Season_Reset_Lambda, 
        Scale_Constant, Brier_Score, LogLoss, Games
    """
    print("\n" + "=" * 60)
    print("Running Sensitivity Analysis (Full Grid Search)")
    print("=" * 60)
    
    # PRE-PROCESS DATA ONCE (this is expensive, so do it once)
    print("Pre-processing game data (one-time cost)...")
    games_prep = (
        raw_data.rename(columns={"Team_FR": "TeamA", "Opp_FR": "TeamB"})
        .sort_values(["Date", "GameID"])[
            ["Date", "GameID", "Season", "TeamA", "TeamB", "A_Home", "B_Home",
             "TeamPoints", "OppPoints", "Outcome"]
        ]
        .reset_index(drop=True)
    )
    
    # Deduplicate - optimized version
    games_prep['GameKey'] = (
        games_prep['Date'].astype(str) + '_' + 
        games_prep['TeamA'].astype(str) + '_' + 
        games_prep['TeamB'].astype(str)
    )
    # Sort teams for consistent keys
    games_prep['TeamPair'] = games_prep[['TeamA', 'TeamB']].apply(
        lambda x: '_'.join(sorted([str(x['TeamA']), str(x['TeamB'])])), axis=1
    )
    games_prep['GameKey'] = games_prep['Date'].astype(str) + '_' + games_prep['TeamPair']
    games_preprocessed = games_prep.drop_duplicates(subset=['GameKey'], keep='first')[
        ["Date", "GameID", "Season", "TeamA", "TeamB", "A_Home", "B_Home",
         "TeamPoints", "OppPoints", "Outcome"]
    ].reset_index(drop=True)
    
    print(f"  Pre-processed {len(games_preprocessed)} unique games")
    
    # Define parameter ranges to test
    k_factors = [15, 20, 25, 30]
    home_advantages = [40, 50, 55, 60, 70]
    season_resets = [0.0, 0.10, 0.15, 0.20]
    scale_constants = [350, 400, 450]
    
    # Calculate total combinations
    total_combinations = len(k_factors) * len(home_advantages) * len(season_resets) * len(scale_constants)
    
    print(f"\nTesting ALL parameter combinations...")
    print(f"  K-factors: {k_factors}")
    print(f"  Home advantages: {home_advantages}")
    print(f"  Season resets: {season_resets}")
    print(f"  Scale constants: {scale_constants}")
    print(f"  Total combinations: {total_combinations}")
    print(f"\nStarting grid search...\n")
    
    results = []
    current = 0
    import time
    start_time = time.time()
    
    # Full grid search - test every combination
    import itertools
    for k, hfa, lam, scale in itertools.product(k_factors, home_advantages, season_resets, scale_constants):
        current += 1
        if current % 20 == 0 or current == 1:
            elapsed = time.time() - start_time
            rate = current / elapsed if elapsed > 0 else 0
            remaining = (total_combinations - current) / rate if rate > 0 else 0
            print(f"  [{current}/{total_combinations}] K={k}, HFA={hfa}, Lambda={lam}, Scale={scale} "
                  f"({rate:.1f} combos/sec, ~{remaining:.0f}s remaining)")
        
        # Use optimized function that only calculates scores
        brier, logloss = calculate_elo_scores_only(
            games_preprocessed, k, hfa, lam, scale
        )
        
        results.append({
            'K_Factor': float(k),
            'Home_Advantage': float(hfa),
            'Season_Reset_Lambda': lam,
            'Scale_Constant': float(scale),
            'Brier_Score': brier,
            'LogLoss': logloss,
            'Games': len(games_preprocessed)
        })
    
    results_df = pd.DataFrame(results)
    
    # Sort by Brier score (lower is better)
    results_df = results_df.sort_values('Brier_Score').reset_index(drop=True)
    
    print(f"\n" + "=" * 60)
    print(f"Sensitivity analysis complete!")
    print(f"  Total combinations tested: {len(results_df)}")
    print(f"  Best Brier Score: {results_df['Brier_Score'].min():.4f}")
    print(f"  Best LogLoss: {results_df['LogLoss'].min():.4f}")
    
    # Find best combination
    best_row = results_df.iloc[0]
    print(f"\n  Best combination:")
    print(f"    K={best_row['K_Factor']:.0f}, HFA={best_row['Home_Advantage']:.0f}, "
          f"Lambda={best_row['Season_Reset_Lambda']:.2f}, Scale={best_row['Scale_Constant']:.0f}")
    print(f"    Brier: {best_row['Brier_Score']:.4f}, LogLoss: {best_row['LogLoss']:.4f}")
    
    # Check default parameters
    print(f"\n  Default parameters (K=20, HFA=55, Lambda=0.15, Scale=400):")
    default_row = results_df[
        (results_df['K_Factor'] == 20.0) & 
        (results_df['Home_Advantage'] == 55.0) & 
        (results_df['Season_Reset_Lambda'] == 0.15) &
        (results_df['Scale_Constant'] == 400.0)
    ]
    if not default_row.empty:
        default_brier = default_row.iloc[0]['Brier_Score']
        default_logloss = default_row.iloc[0]['LogLoss']
        default_rank = results_df.index[results_df['Brier_Score'] == default_brier].tolist()[0] + 1
        print(f"    Brier: {default_brier:.4f}, LogLoss: {default_logloss:.4f}")
        print(f"    Rank: {default_rank}/{len(results_df)}")
        print(f"    Percentile: {(1 - (default_rank - 1) / len(results_df)) * 100:.1f}%")
    
    return results_df

@safefig
def plot_sensitivity_analysis_table(sensitivity_results: pd.DataFrame) -> None:
    """
    Create a table visualization of sensitivity analysis results.
    
    Args:
        sensitivity_results: DataFrame from run_sensitivity_analysis
    """
    print("Creating sensitivity analysis table...")
    
    # Create figure with table
    fig = plt.figure(figsize=(16, max(10, len(sensitivity_results) * 0.3)))
    ax = fig.add_subplot(111)
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table - show top 20 results
    display_df = sensitivity_results.head(20).copy()
    
    # Format numbers for display
    display_df['K_Factor'] = display_df['K_Factor'].astype(int)
    display_df['Home_Advantage'] = display_df['Home_Advantage'].astype(int)
    display_df['Season_Reset_Lambda'] = display_df['Season_Reset_Lambda'].round(2)
    display_df['Scale_Constant'] = display_df['Scale_Constant'].astype(int)
    display_df['Brier_Score'] = display_df['Brier_Score'].round(4)
    display_df['LogLoss'] = display_df['LogLoss'].round(4)
    
    # Rename columns for display
    display_df = display_df.rename(columns={
        'K_Factor': 'K',
        'Home_Advantage': 'HFA',
        'Season_Reset_Lambda': 'Lambda',
        'Scale_Constant': 'Scale',
        'Brier_Score': 'Brier',
        'LogLoss': 'LogLoss'
    })
    
    # Create table
    table = ax.table(
        cellText=display_df[['K', 'HFA', 'Lambda', 'Scale', 'Brier', 'LogLoss']].values,
        colLabels=['K', 'HFA', 'Lambda', 'Scale', 'Brier', 'LogLoss'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color code top 5 rows (best performers)
    for i in range(1, min(6, len(display_df) + 1)):
        for j in range(6):
            table[(i, j)].set_facecolor('#e8f5e9')  # Light green
    
    # Highlight default parameters row
    default_idx = display_df[
        (display_df['K'] == 20) & 
        (display_df['HFA'] == 55) & 
        (display_df['Lambda'] == 0.15) &
        (display_df['Scale'] == 400)
    ].index
    
    if len(default_idx) > 0:
        row_idx = display_df.index.get_loc(default_idx[0]) + 1
        for j in range(6):
            table[(row_idx, j)].set_facecolor('#fff9c4')  # Light yellow
    
    # Style header
    for j in range(6):
        table[(0, j)].set_facecolor(config.COLORS['primary'])
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Sensitivity Analysis: Top 20 Parameter Combinations\n(Sorted by Brier Score - Lower is Better)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Save figure
    output_path = config.OUTPUT_DIR / 'sensitivity_analysis_table.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor=config.COLORS['background'])
    print(f"Saved sensitivity analysis table to {output_path}")
    
    # Also save as CSV
    csv_path = config.OUTPUT_DIR / 'sensitivity_analysis_results.csv'
    sensitivity_results.to_csv(csv_path, index=False)
    print(f"Saved sensitivity analysis results to {csv_path}")
    
    plt.tight_layout()
    if config.SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

def build_upset_map(
    elo_games: pd.DataFrame,
    elo_bin_edges: list[int] | np.ndarray,
    min_bin_n: int = 30
) -> pd.DataFrame:
    """
    Compute upset statistics over Elo-difference bins by season.

    Definitions:
        - Underdog: team with lower pre-game Elo.
        - Upset: underdog wins (ties excluded).
        - Expected underdog win prob: min(A_Exp, B_Exp).

    Outputs:
        For each (Season, EloDiff bin):
            UpsetRate  = mean(upset)
            ExpUpset   = mean(expected underdog win prob)
            Gap        = UpsetRate − ExpUpset  (positive ⇒ more chaos than expected)
            N          = games in bin (after filters)
    """
    print("Building upset map...")
    g = elo_games.copy()
    
    # Calculate effective Elo (pre-game + home field advantage) for underdog definition
    # This aligns underdog definition with the actual win probabilities used
    hfa_a = np.where((g["A_Home"] == 1) & (g["B_Home"] == 0), 
                    config.HOME_FIELD_ADVANTAGE, 0.0)
    hfa_b = np.where((g["B_Home"] == 1) & (g["A_Home"] == 0), 
                    config.HOME_FIELD_ADVANTAGE, 0.0)
    a_eff_elo = g["A_Pre"] + hfa_a
    b_eff_elo = g["B_Pre"] + hfa_b
    
    # Define underdog as team with lower effective Elo (exclude ties)
    lower_is_A = a_eff_elo < b_eff_elo
    equal_pre  = np.isclose(a_eff_elo, b_eff_elo)
    a_win      = np.isclose(g["A_Act"], 1.0)
    tie        = np.isclose(g["A_Act"], 0.5)

    upset = np.where(equal_pre, np.nan, np.where(lower_is_A, a_win, ~a_win)).astype(float)
    upset = np.where(tie, np.nan, upset)

    # Expected upset probability = min(A_Exp, B_Exp)
    lower_exp = np.minimum(g["A_Exp"], g["B_Exp"]).astype(float)
    lower_exp = np.where(equal_pre | tie, np.nan, lower_exp)

    g["EloDiffAbs"] = np.abs(g["A_Pre"] - g["B_Pre"]).astype(float)
    g["Bin"] = pd.cut(g["EloDiffAbs"], bins=elo_bin_edges, include_lowest=True, right=False)

    up = pd.DataFrame({
        "Season": g["Season"].to_numpy(),
        "Bin": g["Bin"].to_numpy(),
        "Upset": upset,
        "Exp": lower_exp
    })
    up = up[up["Upset"].notna() & up["Exp"].notna()]

    heat = (
        up.groupby(["Season","Bin"], observed=True)
          .agg(
              UpsetRate=("Upset","mean"),
              ExpUpset=("Exp","mean"),
              N=("Upset","size")
          )
          .reset_index()
    )
    heat = heat[heat["N"] >= min_bin_n].reset_index(drop=True)
    heat["Gap"] = heat["UpsetRate"] - heat["ExpUpset"]
    print(f"Built upset map with {len(heat)} records")
    return heat

def build_sos(season_team_summary: pd.DataFrame, timeline: pd.DataFrame, elo_games: pd.DataFrame, weighting: str = "equal") -> pd.DataFrame:
    """
    Compute real Strength of Schedule (SoS) with multiple metrics.
    
    Calculates four SoS variants:
    - SoS_PreMean: Mean opponent ELO before game
    - SoS_EndMean: Mean opponent end-of-season ELO
    - SoS_PreWeighted: Mean opponent ELO (weighted by home/away)
    - SoS_EndWeighted: Mean opponent end-of-season ELO (weighted by home/away)
    
    Args:
        season_team_summary: DataFrame with season team summaries (for end-of-season ELO)
        timeline: DataFrame with team timeline data (for opponent info)
        elo_games: DataFrame with ELO game data (for pre-game opponent ELO)
        weighting: "equal" or "home_adj" (affects home/away weighting)
        
    Returns:
        DataFrame with [Season, Team, SoS_PreMean, SoS_EndMean, SoS_PreWeighted, SoS_EndWeighted, Games]
    """
    print("Building strength of schedule...")
    
    # Get end-of-season ELO for each team-season
    end_elo = season_team_summary.set_index(['Season', 'Team'])['Elo_Post'].to_dict()
    
    # Create opponent ELO lookup from elo_games
    # For each game, we need to know opponent's pre-game ELO
    opponent_pre_elo = {}
    for _, game in elo_games.iterrows():
        season = int(game['Season'])
        date = game['Date']
        team_a = game['TeamA']
        team_b = game['TeamB']
        
        # Store opponent pre-game ELO for both teams
        key_a = (season, team_a, date, team_b)
        key_b = (season, team_b, date, team_a)
        opponent_pre_elo[key_a] = game['B_Pre']  # Team A's opponent (B) pre-game ELO
        opponent_pre_elo[key_b] = game['A_Pre']  # Team B's opponent (A) pre-game ELO
    
    sos_data = []
    
    for season in sorted(timeline['Season'].unique()):
        season_data = timeline[timeline['Season'] == season]
        season_games = elo_games[elo_games['Season'] == season]
        
        for team in season_data['Team'].unique():
            team_season = season_data[season_data['Team'] == team].sort_values('Date')
            team_games = season_games[(season_games['TeamA'] == team) | (season_games['TeamB'] == team)]
            
            if team_season.empty:
                continue
            
            # Collect opponent ELOs and home/away weights
            opp_pre_elos = []
            opp_end_elos = []
            home_weights = []
            
            for _, game_row in team_season.iterrows():
                opponent = game_row['Opponent']
                date = game_row['Date']
                is_home = game_row['Home']
                
                # Get opponent pre-game ELO
                lookup_key = (season, team, date, opponent)
                if lookup_key in opponent_pre_elo:
                    opp_pre = opponent_pre_elo[lookup_key]
                else:
                    # Fallback: try to get from elo_games directly
                    opp_game = team_games[
                        ((team_games['TeamA'] == team) & (team_games['TeamB'] == opponent)) |
                        ((team_games['TeamB'] == team) & (team_games['TeamA'] == opponent))
                    ]
                    if not opp_game.empty:
                        opp_game_dates = pd.to_datetime(opp_game['Date'])
                        date_match = opp_game_dates == pd.to_datetime(date)
                        if date_match.any():
                            opp_game_row = opp_game[date_match].iloc[0]
                            if team == opp_game_row['TeamA']:
                                opp_pre = opp_game_row['B_Pre']
                            else:
                                opp_pre = opp_game_row['A_Pre']
                        else:
                            opp_pre = config.START_ELO  # Default fallback
                    else:
                        opp_pre = config.START_ELO  # Default fallback
                
                # Get opponent end-of-season ELO
                opp_end = end_elo.get((season, opponent), config.START_ELO)
                
                # Home/away weight (1.0 for home, 1.1-1.2 for away, or vice versa)
                # Standard: away games are harder, so weight them more
                if weighting == "home_adj":
                    weight = 1.15 if not is_home else 1.0  # Away games weighted 15% more
                else:
                    weight = 1.0  # Equal weighting
                
                opp_pre_elos.append(opp_pre)
                opp_end_elos.append(opp_end)
                home_weights.append(weight)
            
            if len(opp_pre_elos) == 0:
                continue
            
            # Calculate four SoS metrics
            opp_pre_elos = np.array(opp_pre_elos)
            opp_end_elos = np.array(opp_end_elos)
            home_weights = np.array(home_weights)
            
            sos_pre_mean = np.mean(opp_pre_elos)
            sos_end_mean = np.mean(opp_end_elos)
            # Use proper weighted average (normalized by sum of weights) for comparability
            sos_pre_weighted = np.average(opp_pre_elos, weights=home_weights)
            sos_end_weighted = np.average(opp_end_elos, weights=home_weights)
            
            sos_data.append({
                'Season': season,
                'Team': team,
                'SoS_PreMean': sos_pre_mean,
                'SoS_EndMean': sos_end_mean,
                'SoS_PreWeighted': sos_pre_weighted,
                'SoS_EndWeighted': sos_end_weighted,
                'Games': len(opp_pre_elos)
            })
    
    sos_df = pd.DataFrame(sos_data)
    
    # Merge with season team summary
    result = season_team_summary.merge(sos_df, on=['Season', 'Team'], how='left')
    
    print(f"Built SoS with {len(result)} records")
    print(f"  SoS_PreMean range: {result['SoS_PreMean'].min():.1f} - {result['SoS_PreMean'].max():.1f}")
    print(f"  SoS_EndMean range: {result['SoS_EndMean'].min():.1f} - {result['SoS_EndMean'].max():.1f}")
    
    return result

def build_parity_metrics_clean(season_team_summary: pd.DataFrame, elo_games: pd.DataFrame) -> pd.DataFrame:
    """
    Build comprehensive parity metrics with z-scores and composite index.
    
    This replicates the original script's parity analysis with four components:
    - Dispersion (lower std = more parity)
    - Rank Stability (lower Spearman correlation = more parity) 
    - Upset Gap (higher actual vs expected upsets = more parity)
    - Forecast Entropy (higher entropy = more parity)
    
    Args:
        season_team_summary: DataFrame with season-end team summaries
        elo_games: DataFrame with ELO game data
        
    Returns:
        DataFrame with parity metrics including z-scores and composite index
    """
    print("Building comprehensive parity metrics...")
    
    cfg = config.get_plot_config('parity')
    
    # Calculate basic parity metrics by season
    parity_data = []
    
    for season in sorted(season_team_summary['Season'].unique()):
        season_data = season_team_summary[season_team_summary['Season'] == season]
        
        if len(season_data) < 2:
            continue
        
        # Get start and end-of-season ELO ratings
        # Using groupby ensures same teams in same order for both series
        season_start = season_data.groupby('Team')['Elo_Pre'].first()
        season_end = season_data.groupby('Team')['Elo_Post'].last()
        
        # Ensure both series are aligned (same teams, same order)
        # This handles edge cases where a team might appear in one but not the other
        common_teams = season_start.index.intersection(season_end.index)
        if len(common_teams) < 2:
            rank_stability = np.nan
        else:
            # Align both series to common teams
            start_aligned = season_start.loc[common_teams]
            end_aligned = season_end.loc[common_teams]
            # Spearman correlation: pandas converts values to ranks internally
            # Uses average rank method for ties (standard approach)
            rank_stability = start_aligned.corr(end_aligned, method='spearman')
        
        # 1. Dispersion (standard deviation of ELO ratings)
        dispersion_std = season_end.std()
        
        # 3. Upset Gap (actual upsets vs expected)
        season_games = elo_games[elo_games['Season'] == season].copy()
        if not season_games.empty:
            # Calculate effective Elo (pre-game + home field advantage) for underdog definition
            # This aligns underdog definition with the actual win probabilities used
            hfa_a = np.where((season_games['A_Home'] == 1) & (season_games['B_Home'] == 0), 
                            config.HOME_FIELD_ADVANTAGE, 0.0)
            hfa_b = np.where((season_games['B_Home'] == 1) & (season_games['A_Home'] == 0), 
                            config.HOME_FIELD_ADVANTAGE, 0.0)
            a_eff_elo = season_games['A_Pre'] + hfa_a
            b_eff_elo = season_games['B_Pre'] + hfa_b
            
            # Underdog is team with lower effective Elo
            season_games['Underdog'] = a_eff_elo < b_eff_elo
            season_games['Underdog_Won'] = (
                (season_games['Underdog'] & (season_games['A_Act'] == 1)) |
                (~season_games['Underdog'] & (season_games['B_Act'] == 1))
            )
            expected_upsets = season_games['A_Exp'].where(season_games['Underdog'], season_games['B_Exp']).sum()
            actual_upsets = season_games['Underdog_Won'].sum()
            # Normalize by number of games for cross-season comparability
            games = len(season_games)
            upset_gap = (actual_upsets - expected_upsets) / games if games > 0 else np.nan
        else:
            upset_gap = np.nan
        
        # 4. Forecast Entropy (average entropy of win probabilities)
        if not season_games.empty:
            # Calculate entropy for each game: -p*log(p) - (1-p)*log(1-p)
            p = season_games['A_Exp']
            entropy = -p * np.log(p + 1e-12) - (1-p) * np.log(1-p + 1e-12)
            forecast_entropy = entropy.mean()
        else:
            forecast_entropy = np.nan
        
        parity_data.append({
            'Season': season,
            'Teams': len(season_end),
            'Dispersion_STD': dispersion_std,
            'RankStability_Spearman': rank_stability,
            'Upset_Gap': upset_gap,
            'Forecast_Entropy': forecast_entropy,
            'ELO_Mean': season_end.mean(),
            'ELO_Range': season_end.max() - season_end.min(),
            'Top_Team': season_end.idxmax(),
            'Bottom_Team': season_end.idxmin()
        })
    
    pm = pd.DataFrame(parity_data)
    
    # Build direction-aligned component scores (higher = more parity)
    # Only compute columns that user enabled; others are NaN
    pm['Z_Dispersion'] = -_z(pm['Dispersion_STD']) if cfg['use_dispersion'] else np.nan
    pm['Z_RankStab'] = -_z(pm['RankStability_Spearman']) if cfg['use_rank_stability'] else np.nan
    pm['Z_UpsetGap'] = _z(pm['Upset_Gap']) if cfg['use_upset_gap'] else np.nan
    pm['Z_Entropy'] = _z(pm['Forecast_Entropy']) if cfg['use_entropy'] else np.nan
    
    # Weights: use config weights, collect enabled components and renormalize to 1.0
    weights = {}
    if cfg['use_dispersion']: weights['Z_Dispersion'] = config.PARITY_WEIGHTS['dispersion']
    if cfg['use_rank_stability']: weights['Z_RankStab'] = config.PARITY_WEIGHTS['rank_stab']
    if cfg['use_upset_gap']: weights['Z_UpsetGap'] = config.PARITY_WEIGHTS['upset_gap']
    if cfg['use_entropy']: weights['Z_Entropy'] = config.PARITY_WEIGHTS['entropy']
    
    wsum = sum(weights.values()) if weights else 0.0
    if wsum <= 0:
        pm['Parity_Index'] = np.nan
        pm['Parity_Score'] = np.nan
    else:
        for k in list(weights.keys()):
            weights[k] = weights[k] / wsum
        # Weighted average across enabled z-scores
        zmat = np.column_stack([pm[k].to_numpy() for k in weights.keys()])
        wvec = np.array([weights[k] for k in weights.keys()])
        pm['Parity_Index'] = np.nanmean(zmat * wvec, axis=1)
        
        # Rescale to 0-100 "Parity Score" anchored to reference era
        anchor_seasons = pm[
            (pm['Season'] >= config.PARITY_ANCHOR_ERA_START) & 
            (pm['Season'] <= config.PARITY_ANCHOR_ERA_END)
        ]
        if not anchor_seasons.empty and anchor_seasons['Parity_Index'].notna().any():
            anchor_mean = anchor_seasons['Parity_Index'].mean()
            anchor_std = anchor_seasons['Parity_Index'].std()
            if anchor_std > 0:
                # Rescale: anchor era = 50, then scale by anchor std
                pm['Parity_Score'] = 50 + (pm['Parity_Index'] - anchor_mean) / anchor_std * 10
                # Clip to min/max range
                pm['Parity_Score'] = pm['Parity_Score'].clip(
                    config.PARITY_SCALE_MIN, config.PARITY_SCALE_MAX
                )
            else:
                pm['Parity_Score'] = 50.0
        else:
            # Fallback: simple min-max scaling
            pmin = pm['Parity_Index'].min()
            pmax = pm['Parity_Index'].max()
            if pmax > pmin:
                pm['Parity_Score'] = config.PARITY_SCALE_MIN + \
                    (pm['Parity_Index'] - pmin) / (pmax - pmin) * \
                    (config.PARITY_SCALE_MAX - config.PARITY_SCALE_MIN)
            else:
                pm['Parity_Score'] = 50.0
    
    print(f"Built comprehensive parity metrics for {len(pm)} seasons")
    
    return pm

def export_comprehensive_data_csv(timeline: pd.DataFrame, elo_games: pd.DataFrame, 
                                   season_team_summary: pd.DataFrame, luck_index: pd.DataFrame,
                                   sos_data: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
    """
    Export comprehensive CSV with all calculated data for each team-game.
    
    Combines:
    - Game-level ELO data (pre/post game ELO, changes, expected probabilities)
    - Team timeline data (points, outcomes, home/away)
    - Season-level aggregations (volatility, dynasty score, rankings)
    - Luck metrics (luck index, close game luck, Pythagorean luck)
    - Strength of schedule metrics (all four variants)
    
    Args:
        timeline: DataFrame with team timeline data
        elo_games: DataFrame with ELO game data
        season_team_summary: DataFrame with season team summaries
        luck_index: DataFrame with luck index data
        sos_data: DataFrame with strength of schedule data
        output_path: Path to save CSV (defaults to Outputs/comprehensive_nfl_data.csv)
        
    Returns:
        DataFrame with all comprehensive data
    """
    print("Exporting comprehensive data to CSV...")
    
    # Start with timeline as base (one row per team per game)
    comprehensive = timeline.copy()
    
    # Add game-level ELO data from elo_games
    # For each game in timeline, we need to get the corresponding row from elo_games
    elo_game_lookup = {}
    for _, game in elo_games.iterrows():
        date = game['Date']
        team_a = game['TeamA']
        team_b = game['TeamB']
        
        # Create lookup keys for both team perspectives
        key_a = (date, team_a, team_b)
        key_b = (date, team_b, team_a)
        
        elo_game_lookup[key_a] = {
            'GameID': game['GameID'],
            'Opp_Pre_ELO': game['B_Pre'],
            'Opp_Post_ELO': game['B_Post'],
            'Opp_Expected': game['B_Exp'],
            'Opp_Actual': game['B_Act'],
            'Opp_Delta': game['B_Delta'],
            'Opp_Points': game['B_Points'],
            'PointDiff': game['PointDiff'],
            'MOV_Factor': game.get('MOV_Factor', 1.0),
            'LeagueMean': game.get('LeagueMean', None),
        }
        
        elo_game_lookup[key_b] = {
            'GameID': game['GameID'],
            'Opp_Pre_ELO': game['A_Pre'],
            'Opp_Post_ELO': game['A_Post'],
            'Opp_Expected': game['A_Exp'],
            'Opp_Actual': game['A_Act'],
            'Opp_Delta': game['A_Delta'],
            'Opp_Points': game['A_Points'],
            'PointDiff': -game['PointDiff'],  # Reverse for team B perspective
            'MOV_Factor': game.get('MOV_Factor', 1.0),
            'LeagueMean': game.get('LeagueMean', None),
        }
    
    # Add game-level data to comprehensive
    game_data = []
    for _, row in comprehensive.iterrows():
        key = (row['Date'], row['Team'], row['Opponent'])
        if key in elo_game_lookup:
            game_data.append(elo_game_lookup[key])
        else:
            game_data.append({k: None for k in ['GameID', 'Opp_Pre_ELO', 'Opp_Post_ELO', 
                                                  'Opp_Expected', 'Opp_Actual', 'Opp_Delta',
                                                  'Opp_Points', 'PointDiff', 'MOV_Factor', 'LeagueMean']})
    
    game_df = pd.DataFrame(game_data)
    comprehensive = pd.concat([comprehensive, game_df], axis=1)
    
    # Add season-level aggregations (volatility, dynasty, rankings)
    season_cols = ['Volatility', 'DynastyScore', 'Start_Rank', 'End_Rank', 
                   'Max_ELO', 'Min_ELO', 'ELO_Range', 'Elo_Pre', 'Elo_Post']
    season_data = season_team_summary[['Season', 'Team'] + season_cols].copy()
    season_data = season_data.rename(columns={
        'Elo_Pre': 'Season_Start_ELO',
        'Elo_Post': 'Season_End_ELO'
    })
    comprehensive = comprehensive.merge(season_data, on=['Season', 'Team'], how='left', suffixes=('', '_season'))
    
    # Add luck metrics
    luck_cols = ['ActualWins', 'ExpWins', 'Luck_Index', 'Luck_per_Game',
                 'Close_ActualWins', 'Close_ExpWins', 'Close_Games', 'Close_Luck_Index', 'Close_Luck_per_Game',
                 'Points_For', 'Points_Against', 'Pyth_ExpWins', 'Pyth_Luck']
    luck_data = luck_index[['Season', 'Team'] + luck_cols].copy()
    comprehensive = comprehensive.merge(luck_data, on=['Season', 'Team'], how='left', suffixes=('', '_luck'))
    
    # Add strength of schedule metrics
    sos_cols = ['SoS_PreMean', 'SoS_EndMean', 'SoS_PreWeighted', 'SoS_EndWeighted']
    # Check if 'Games' column exists (it might be merged into season_team_summary)
    if 'Games' in sos_data.columns:
        sos_cols.append('Games')
    
    sos_subset = sos_data[['Season', 'Team'] + sos_cols].copy()
    if 'Games' in sos_subset.columns:
        sos_subset = sos_subset.rename(columns={'Games': 'SoS_Games'})
    comprehensive = comprehensive.merge(sos_subset, on=['Season', 'Team'], how='left', suffixes=('', '_sos'))
    
    # Calculate cumulative stats up to this game in the season
    comprehensive = comprehensive.sort_values(['Team', 'Season', 'Date'])
    comprehensive['Game_Number'] = comprehensive.groupby(['Team', 'Season']).cumcount() + 1
    
    # Calculate running totals for the season up to this game
    comprehensive['Cumulative_Wins'] = comprehensive.groupby(['Team', 'Season'])['Actual'].transform(
        lambda x: (x == 1.0).cumsum()
    )
    comprehensive['Cumulative_ExpWins'] = comprehensive.groupby(['Team', 'Season'])['Expected'].transform('cumsum')
    comprehensive['Cumulative_Luck'] = comprehensive['Cumulative_Wins'] - comprehensive['Cumulative_ExpWins']
    comprehensive['Cumulative_Points_For'] = comprehensive.groupby(['Team', 'Season'])['Points_For'].transform('cumsum')
    comprehensive['Cumulative_Points_Against'] = comprehensive.groupby(['Team', 'Season'])['Points_Against'].transform('cumsum')
    
    # Reorder columns for better readability
    priority_cols = [
        'Date', 'Season', 'GameID', 'Game_Number',
        'Team', 'Opponent', 'Home',
        'Elo_Pre', 'Elo_Post', 'Elo_Change', 'Expected', 'Actual', 'Outcome',
        'Opp_Pre_ELO', 'Opp_Post_ELO', 'Opp_Expected', 'Opp_Actual', 'Opp_Delta',
        'Points_For', 'Points_Against', 'Opp_Points', 'PointDiff',
        'MOV_Factor', 'LeagueMean',
        'Season_Start_ELO', 'Season_End_ELO', 'Start_Rank', 'End_Rank',
        'Volatility', 'DynastyScore', 'Max_ELO', 'Min_ELO', 'ELO_Range',
        'Cumulative_Wins', 'Cumulative_ExpWins', 'Cumulative_Luck',
        'Cumulative_Points_For', 'Cumulative_Points_Against',
        'ActualWins', 'ExpWins', 'Luck_Index', 'Luck_per_Game',
        'Close_ActualWins', 'Close_ExpWins', 'Close_Games', 'Close_Luck_Index', 'Close_Luck_per_Game',
        'Points_For_luck', 'Points_Against_luck', 'Pyth_ExpWins', 'Pyth_Luck',
        'SoS_PreMean', 'SoS_EndMean', 'SoS_PreWeighted', 'SoS_EndWeighted', 'SoS_Games'
    ]
    
    # Get all columns that exist
    existing_cols = [col for col in priority_cols if col in comprehensive.columns]
    # Add any remaining columns
    remaining_cols = [col for col in comprehensive.columns if col not in existing_cols]
    final_cols = existing_cols + remaining_cols
    
    comprehensive = comprehensive[final_cols]
    
    # Round all float columns to 4 decimal places (ten-thousandths)
    # This prevents Excel from treating long decimals as text
    # Only round float columns, not integer columns (like Game_Number, Games, etc.)
    float_cols = comprehensive.select_dtypes(include=[np.floating]).columns
    for col in float_cols:
        comprehensive[col] = comprehensive[col].round(4)
    
    # Save to CSV
    if output_path is None:
        output_path = config.OUTPUT_DIR / 'comprehensive_nfl_data.csv'
    
    comprehensive.to_csv(output_path, index=False)
    print(f"Exported comprehensive data to {output_path}")
    print(f"  Total records: {len(comprehensive)}")
    print(f"  Columns: {len(comprehensive.columns)}")
    print(f"  Date range: {comprehensive['Date'].min()} to {comprehensive['Date'].max()}")
    
    return comprehensive

def calculate_volatility_and_dynasty(timeline: pd.DataFrame, season_team_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate ELO volatility and dynasty scores for each team-season.
    
    Volatility: Standard deviation of game-to-game ELO changes
    Dynasty Score: Sum of ELO above threshold (config.DYNASTY_THRESHOLD) across games
    
    Args:
        timeline: DataFrame with team ELO timeline
        season_team_summary: DataFrame with season team summaries
        
    Returns:
        DataFrame with [Season, Team, Volatility, DynastyScore, ...]
    """
    print("Calculating volatility and dynasty scores...")
    
    volatility_data = []
    
    for season in sorted(timeline['Season'].unique()):
        season_data = timeline[timeline['Season'] == season]
        
        for team in season_data['Team'].unique():
            team_season = season_data[season_data['Team'] == team].sort_values('Date')
            
            if team_season.empty:
                continue
            
            # Volatility: std of game-to-game ELO changes
            elo_changes = team_season['Elo_Change']
            volatility = elo_changes.std() if len(elo_changes) > 1 else 0.0
            
            # Dynasty score: sum of ELO above threshold across all games
            # Formula: sum(max(0, ELO_post - threshold)) / 100
            # This will produce decimal values because ELO values are floats
            # Example: If ELO = 1151.2345 and threshold = 1150, excess = 1.2345
            # Normalized by 100 to get a more interpretable scale
            elo_post = team_season['Elo_Post'].dropna()  # Remove any NaN values
            threshold = config.DYNASTY_THRESHOLD
            
            # Calculate sum of (ELO - threshold) for all games where ELO > threshold
            if len(elo_post) > 0:
                above_threshold = elo_post[elo_post > threshold]
                if len(above_threshold) > 0:
                    # Sum the excess ELO above threshold, then normalize by 100
                    dynasty_score = (above_threshold - threshold).sum() / 100.0
                else:
                    dynasty_score = 0.0
            else:
                dynasty_score = 0.0
            
            # Alternative: integral of ELO above threshold over time
            # For simplicity, using sum approach above
            
            volatility_data.append({
                'Season': season,
                'Team': team,
                'Volatility': volatility,
                'DynastyScore': dynasty_score,
                'Max_ELO': elo_post.max(),
                'Min_ELO': elo_post.min(),
                'ELO_Range': elo_post.max() - elo_post.min()
            })
    
    vol_df = pd.DataFrame(volatility_data)
    
    # Merge with season team summary
    result = season_team_summary.merge(vol_df, on=['Season', 'Team'], how='left')
    
    print(f"Calculated volatility and dynasty scores for {len(result)} team-seasons")
    print(f"  Average volatility: {result['Volatility'].mean():.1f}")
    print(f"  Average dynasty score: {result['DynastyScore'].mean():.1f}")
    
    return result

def calculate_forecast_sharpness(elo_games: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate forecast sharpness (standard deviation of predicted probabilities).
    
    High sharpness = model makes strong calls (probabilities spread out)
    Low sharpness = model is timid (probabilities cluster around 50%)
    
    Args:
        elo_games: DataFrame with ELO game data
        
    Returns:
        DataFrame with [Season, Sharpness, ...]
    """
    print("Calculating forecast sharpness...")
    
    sharpness_data = []
    
    for season in sorted(elo_games['Season'].unique()):
        season_games = elo_games[elo_games['Season'] == season]
        
        if season_games.empty:
            continue
        
        # Sharpness = std of predicted win probabilities
        probs = season_games['A_Exp']
        sharpness = probs.std()
        
        # Also calculate entropy (already in parity metrics, but useful here too)
        entropy = -probs * np.log(probs + 1e-12) - (1-probs) * np.log(1-probs + 1e-12)
        mean_entropy = entropy.mean()
        
        sharpness_data.append({
            'Season': season,
            'Sharpness': sharpness,
            'Mean_Entropy': mean_entropy,
            'Games': len(season_games)
        })
    
    sharp_df = pd.DataFrame(sharpness_data)
    
    print(f"Calculated sharpness for {len(sharp_df)} seasons")
    print(f"  Average sharpness: {sharp_df['Sharpness'].mean():.3f}")
    
    return sharp_df

def create_division_summary(season_team_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Create division-level summaries with mean ELO, dispersion, and strength rankings.
    Accounts for 2002 realignment (6 divisions pre-2002, 8 divisions post-2002).
    
    Args:
        season_team_summary: DataFrame with season team summaries
        
    Returns:
        DataFrame with division-level metrics
    """
    print("Creating division summaries...")
    
    summary_df = season_team_summary.copy()
    # Apply season-aware division mapping
    summary_df['Division'] = summary_df.apply(
        lambda row: get_team_division(row['Team'], int(row['Season'])), axis=1
    )
    
    # Group by season and division
    division_summary = []
    
    for season in sorted(summary_df['Season'].unique()):
        season_data = summary_df[summary_df['Season'] == season]
        
        for division in season_data['Division'].dropna().unique():
            div_data = season_data[season_data['Division'] == division]
            
            if len(div_data) < 2:
                continue
            
            # Division metrics
            div_mean_elo = div_data['Elo_Post'].mean()
            div_std_elo = div_data['Elo_Post'].std()
            div_range_elo = div_data['Elo_Post'].max() - div_data['Elo_Post'].min()
            
            # Division parity (lower std = more parity within division)
            div_parity = 1.0 / (1.0 + div_std_elo / 100.0)  # Normalized parity score
            
            division_summary.append({
                'Season': season,
                'Division': division,
                'Conference': division.split()[0],  # AFC or NFC
                'Teams': len(div_data),
                'Mean_ELO': div_mean_elo,
                'Std_ELO': div_std_elo,
                'Range_ELO': div_range_elo,
                'Parity_Score': div_parity,
                'Top_Team': div_data.loc[div_data['Elo_Post'].idxmax(), 'Team'],
                'Bottom_Team': div_data.loc[div_data['Elo_Post'].idxmin(), 'Team']
            })
    
    div_df = pd.DataFrame(division_summary)
    
    # Add division strength ranking by season
    div_df['Division_Rank'] = div_df.groupby('Season')['Mean_ELO'].rank(ascending=False, method='dense')
    
    print(f"Created division summaries for {len(div_df)} division-seasons")
    
    return div_df

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

@safefig
def plot_league_tapestry(df: pd.DataFrame, config_dict: Dict = None) -> None:
    """
    Create a league-wide ELO tapestry showing all teams over time.
    
    Args:
        df: DataFrame with ELO data
        config_dict: Plot configuration dictionary
    """
    if not config.is_plot_enabled('league_tapestry'):
        print("League tapestry plot disabled")
        return
    
    if config_dict is None:
        config_dict = config.get_plot_config('league_tapestry')
    
    print("Creating league tapestry plot...")
    
    fig, ax = plt.subplots(figsize=config_dict['figsize'])
    
    # Prepare data
    if config_dict['use_rolling']:
        elo_col = 'Elo_RollMean'
        df_plot = add_rolling_features(df, config_dict['window'])
    else:
        elo_col = 'Elo'
        df_plot = df
    
    # Plot individual teams
    for team in df_plot['Franchise'].unique():
        team_data = df_plot[df_plot['Franchise'] == team].sort_values('Date')
        ax.plot(team_data['Date'], team_data[elo_col], 
               alpha=config_dict['line_alpha'], 
               linewidth=config_dict['team_linewidth'],
               color='gray')
    
    # Plot league average
    league_avg = df_plot.groupby('Date')[elo_col].mean()
    ax.plot(league_avg.index, league_avg.values,
           linewidth=config_dict['league_linewidth'],
           color='red', label='League Average')
    
    # Formatting
    ax.set_title('NFL ELO Tapestry - All Teams Over Time', fontsize=16, fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('ELO Rating')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    
    # Save figure
    output_path = config.OUTPUT_DIR / 'league_tapestry.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved league tapestry to {output_path}")
    
    if config.SHOW_PLOTS:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory

@safefig
def plot_team_small_multiples(team_monthly_elo: pd.DataFrame, config_dict: Dict = None) -> None:
    """
    Create small multiples plot showing individual team trajectories.
    
    Args:
        team_monthly_elo: DataFrame with monthly aggregated ELO data
        config_dict: Plot configuration dictionary
    """
    if not config.is_plot_enabled('team_small_multiples'):
        print("Team small multiples plot disabled")
        return
    
    if config_dict is None:
        config_dict = config.get_plot_config('team_small_multiples')
    
    print("Creating team small multiples plot...")
    
    # Prepare data - use the same logic as original script
    df = team_monthly_elo.copy()
    if config_dict['use_rolling']:
        # Check if rolling features exist, if not create them
        if 'Elo_Mean_RollMean' not in df.columns:
            df = add_rolling_features(df, config_dict['window'])
        ycol = 'Elo_Mean_RollMean'
    else:
        ycol = 'Elo_Mean'
    
    # Select teams
    if config_dict['teams'] is None:
        teams = sorted(df['Franchise'].unique().tolist())
    else:
        teams = config_dict['teams']
    
    n_teams = len(teams)
    n_cols = config_dict['cols']
    n_rows = (n_teams + n_cols - 1) // n_cols
    
    # Create figure with shared axes for standardization
    fig, axes = plt.subplots(n_rows, n_cols, 
                           figsize=(n_cols * config_dict['fig_cell_w'], 
                                  n_rows * config_dict['fig_cell_h']),
                           sharex=True, sharey=True)
    
    # Ensure axes is always 2D
    axes = np.atleast_2d(axes)
    
    # Calculate global y-axis limits for standardization
    ymin = float(df[ycol].min()) - 10.0
    ymax = float(df[ycol].max()) + 10.0
    
    # Plot each team
    for i, team in enumerate(teams):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        
        # Apply custom color scheme to each subplot
        config.apply_color_scheme(ax)
        
        g = df[df['Franchise'] == team].sort_values('YearMonth')
        
        if not g.empty:
            # Use team color if available, otherwise use series color
            team_color = config.get_team_color(team) if team in config.COLORS['team_colors'] else config.get_series_color(i)
            ax.plot(g['YearMonth'], g[ycol], linewidth=1.5, color=team_color, alpha=0.8)
            
            # Add baseline if requested
            if config_dict['show_baseline']:
                ax.axhline(y=config_dict['baseline_value'], 
                          color=config.COLORS['accent'], linestyle='--', alpha=0.6, linewidth=1.0)
            
            # Add realignment marker if requested
            if config_dict['show_realignment']:
                realign_date = pd.to_datetime(config_dict['realignment_date'])
                ax.axvline(x=realign_date, color=config.COLORS['warning'], 
                          linestyle=':', alpha=0.5, linewidth=1.0)
        
        ax.set_title(team, fontsize=10, fontweight='bold', color=config.COLORS['text_primary'])
        ax.set_ylim([ymin, ymax])  # Standardize y-axis limits
        
    # Hide empty subplots
    for j in range(i + 1, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r, c].axis('off')
    
    # Format x-axis labels for all subplots (since they share x-axis)
    import matplotlib.dates as mdates
    
    # Get the bottom row of subplots to format x-axis
    bottom_row = axes[-1, :] if n_rows > 1 else [axes[0, 0]]
    
    for ax in bottom_row:
        if ax.get_visible():  # Only format visible axes
            # Set smaller font size for year labels
            ax.tick_params(axis='x', labelsize=7)
            
            # Format years as 2-digit (e.g., 1980 -> '80)
            ax.xaxis.set_major_locator(mdates.YearLocator(5))  # Every 5 years
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%y'))  # 2-digit year
            
            # Rotate labels slightly for better fit
            ax.tick_params(axis='x', rotation=45)
    
    title_suffix = f"Rolling {config_dict['window']}m" if config_dict['use_rolling'] else "Monthly Mean"
    plt.suptitle(f'Team ELO Timelines ({title_suffix})', fontsize=14, fontweight='bold', y=1.02, color=config.COLORS['text_primary'])
    plt.tight_layout()
    
    # Save figure
    output_path = config.OUTPUT_DIR / 'team_small_multiples.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=config.COLORS['background'])
    print(f"Saved team small multiples to {output_path}")
    
    if config.SHOW_PLOTS:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory

@safefig
def plot_season_ladder(timeline: pd.DataFrame, config_dict: Dict = None) -> None:
    """
    Create a season ladder showing team rankings at season end.
    
    Args:
        timeline: DataFrame with team ELO timeline
        config_dict: Plot configuration dictionary
    """
    if not config.is_plot_enabled('season_ladder'):
        print("Season ladder plot disabled")
        return
    
    if config_dict is None:
        config_dict = config.get_plot_config('season_ladder')
    
    print(f"Creating season ladder for {config_dict['season']}...")
    
    # Get season end ratings
    season_data = timeline[timeline['Season'] == config_dict['season']]
    season_end = season_data.groupby('Team')['Elo_Post'].last().sort_values(ascending=False)
    
    if config_dict['top_n']:
        season_end = season_end.head(config_dict['top_n'])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Apply custom color scheme
    config.apply_color_scheme(ax)
    
    # Create horizontal bar plot with team colors
    y_pos = np.arange(len(season_end))
    colors = [config.get_team_color(team) for team in season_end.index]
    bars = ax.barh(y_pos, season_end.values, color=colors, alpha=0.8, edgecolor=config.COLORS['border'], linewidth=0.5)
    
    # Add value labels
    for i, (team, rating) in enumerate(season_end.items()):
        ax.text(rating + 5, i, f'{rating:.0f}', va='center', fontweight='bold', color=config.COLORS['text_primary'])
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(season_end.index, fontweight='medium')
    ax.set_xlabel('ELO Rating', fontweight='medium')
    ax.set_title(f'NFL Season {config_dict["season"]} Final Rankings', 
                fontsize=16, fontweight='bold', color=config.COLORS['text_primary'])
    
    # Invert y-axis to show highest at top
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    # Save figure
    output_path = config.OUTPUT_DIR / f'season_ladder_{config_dict["season"]}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=config.COLORS['background'])
    print(f"Saved season ladder to {output_path}")
    
    if config.SHOW_PLOTS:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory

@safefig
def plot_league_tapestry(team_monthly_elo: pd.DataFrame, config_dict: Dict = None) -> None:
    """
    Create a league-wide ELO tapestry showing all teams over time.
    
    Args:
        team_monthly_elo: DataFrame with monthly aggregated ELO data
        config_dict: Plot configuration dictionary
    """
    if not config.is_plot_enabled('league_tapestry'):
        print("League tapestry plot disabled")
        return
    
    if config_dict is None:
        config_dict = config.get_plot_config('league_tapestry')
    
    print("Creating league tapestry plot...")
    
    # Prepare data - use the same logic as original script
    df = team_monthly_elo.copy()
    if config_dict['use_rolling']:
        # Check if rolling features exist, if not create them
        if 'Elo_Mean_RollMean' not in df.columns:
            df = add_rolling_features(df, config_dict['window'])
        ycol = 'Elo_Mean_RollMean'
    else:
        ycol = 'Elo_Mean'
    
    # Calculate league average ELO
    # Note: In a pure zero-sum system, this should stay at 1000
    # However, season resets (15% regression to mean) can cause slight drift
    # We calculate as mean of team ELOs per month, then apply smoothing for stability
    league = (
        df.groupby('YearMonth', as_index=False)[ycol]
        .mean()
        .rename(columns={ycol: 'LeagueMean'})
        .sort_values('YearMonth')
    )
    
    # Apply rolling smoothing to league average to reduce bumps
    # Use a 6-month rolling window to smooth out monthly variations and season transitions
    league['LeagueMean'] = league['LeagueMean'].rolling(window=6, min_periods=1, center=True).mean()
    
    # The league average should theoretically be 1000, but season resets cause drift
    # This is expected behavior with season resets enabled
    
    fig, ax = plt.subplots(figsize=config_dict['figsize'])
    
    # Apply custom color scheme
    config.apply_color_scheme(ax)
    
    # Plot individual teams with custom colors
    for i, (_, g) in enumerate(df.groupby('Franchise')):
        team_color = config.get_team_color(g['Franchise'].iloc[0]) if g['Franchise'].iloc[0] in config.COLORS['team_colors'] else config.COLORS['secondary']
        ax.plot(g['YearMonth'], g[ycol], 
               alpha=config_dict['line_alpha'], 
               linewidth=config_dict['team_linewidth'],
               color=team_color)
    
    # Plot league average with custom styling
    ax.plot(league['YearMonth'], league['LeagueMean'],
           linewidth=config_dict['league_linewidth'],
           color=config.COLORS['accent'], label='League Average', alpha=0.95)
    
    # Formatting
    title_suffix = f"Rolling {config_dict['window']}m" if config_dict['use_rolling'] else "Monthly Mean"
    ax.set_title(f'League ELO Tapestry ({title_suffix})', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Year', fontsize=12, fontweight='medium')
    ax.set_ylabel('ELO Rating', fontsize=12, fontweight='medium')
    ax.legend(framealpha=0.9, edgecolor=config.COLORS['border'])
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    
    # Save figure
    output_path = config.OUTPUT_DIR / 'league_tapestry.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=config.COLORS['background'])
    print(f"Saved league tapestry to {output_path}")
    
    if config.SHOW_PLOTS:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory

@safefig
def plot_team_summary_stats(summary_stats: pd.DataFrame, config_dict: Dict = None) -> None:
    """
    Create a visualization of team summary statistics.
    
    Args:
        summary_stats: DataFrame with team summary statistics
        config_dict: Plot configuration dictionary
    """
    if not config.is_plot_enabled('team_summary'):
        print("Team summary plot disabled")
        return
    
    if config_dict is None:
        config_dict = config.get_plot_config('team_summary')
    
    print("Creating team summary statistics plot...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Apply color scheme to all subplots
    for ax in [ax1, ax2, ax3, ax4]:
        config.apply_color_scheme(ax)
    
    # Current ELO rankings
    top_teams = summary_stats.head(config_dict.get('top_n', 16))
    colors = [config.get_team_color(team) if team in config.COLORS['team_colors'] else config.COLORS['primary'] for team in top_teams['Team']]
    ax1.barh(range(len(top_teams)), top_teams['Current_ELO'], color=colors, alpha=0.8, edgecolor=config.COLORS['border'], linewidth=0.5)
    ax1.set_yticks(range(len(top_teams)))
    ax1.set_yticklabels(top_teams['Team'], fontweight='medium')
    ax1.set_xlabel('Current ELO Rating', fontweight='medium')
    ax1.set_title('Current ELO Rankings (Top 16)', fontweight='bold', fontsize=12)
    
    # Win percentage vs ELO
    colors = [config.get_team_color(team) if team in config.COLORS['team_colors'] else config.COLORS['success'] for team in summary_stats['Team']]
    ax2.scatter(summary_stats['Win_Pct'], summary_stats['Current_ELO'], 
               alpha=0.8, s=80, c=colors, edgecolors=config.COLORS['primary'], linewidth=0.5)
    ax2.set_xlabel('Win Percentage', fontweight='medium')
    ax2.set_ylabel('Current ELO Rating', fontweight='medium')
    ax2.set_title('Win Percentage vs ELO Rating', fontweight='bold', fontsize=12)
    
    # Add team labels based on toggle
    if config_dict.get('show_team_names', True):
        # Show all team names
        for _, row in summary_stats.iterrows():
            ax2.annotate(row['Team'], (row['Win_Pct'], row['Current_ELO']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8, 
                        color=config.COLORS['text_secondary'], fontweight='medium')
    else:
        # Show only extreme points (original behavior)
        for _, row in summary_stats.iterrows():
            if row['Win_Pct'] > 0.7 or row['Current_ELO'] > 1600:
                ax2.annotate(row['Team'], (row['Win_Pct'], row['Current_ELO']), 
                            xytext=(5, 5), textcoords='offset points', fontsize=8, 
                            color=config.COLORS['text_secondary'], fontweight='medium')
    
    # ELO range (peak - low)
    summary_stats['ELO_Range'] = summary_stats['Peak_ELO'] - summary_stats['Low_ELO']
    top_range = summary_stats.nlargest(12, 'ELO_Range')
    colors = [config.get_team_color(team) if team in config.COLORS['team_colors'] else config.COLORS['warning'] for team in top_range['Team']]
    ax3.barh(range(len(top_range)), top_range['ELO_Range'], color=colors, alpha=0.8, edgecolor=config.COLORS['border'], linewidth=0.5)
    ax3.set_yticks(range(len(top_range)))
    ax3.set_yticklabels(top_range['Team'], fontweight='medium')
    ax3.set_xlabel('ELO Range (Peak - Low)', fontweight='medium')
    ax3.set_title('Teams with Largest ELO Ranges', fontweight='bold', fontsize=12)
    
    # Average ELO (sustained performance over all games)
    top_avg = summary_stats.nlargest(12, 'Avg_ELO')
    colors = [config.get_team_color(team) if team in config.COLORS['team_colors'] else config.COLORS['info'] for team in top_avg['Team']]
    ax4.barh(range(len(top_avg)), top_avg['Avg_ELO'], color=colors, alpha=0.8, edgecolor=config.COLORS['border'], linewidth=0.5)
    ax4.set_yticks(range(len(top_avg)))
    ax4.set_yticklabels(top_avg['Team'], fontweight='medium')
    ax4.set_xlabel('Average ELO Rating', fontweight='medium')
    ax4.set_title('Highest Average ELO (All-Time)', fontweight='bold', fontsize=12)
    
    plt.suptitle('NFL Team Summary Statistics', fontsize=16, fontweight='bold', color=config.COLORS['text_primary'])
    plt.tight_layout()
    
    # Save figure
    output_path = config.OUTPUT_DIR / 'team_summary_stats.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=config.COLORS['background'])
    print(f"Saved team summary stats to {output_path}")
    
    if config.SHOW_PLOTS:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory

@safefig
def plot_biggest_swings(swings: pd.DataFrame, config_dict: Dict = None) -> None:
    """
    Create a table visualization of the top 10 upsets of all time.
    
    Args:
        swings: DataFrame with biggest swings data
        config_dict: Plot configuration dictionary
    """
    if not config.is_plot_enabled('biggest_swings'):
        print("Biggest swings plot disabled")
        return
    
    print("Creating top 10 upsets table...")
    
    # Filter for the biggest upsets (largest positive ELO changes)
    upsets = swings[swings['Swing_Type'] == 'Gain'].head(10).copy()
    
    if upsets.empty:
        print("No upset data available")
        return
    
    # Sort by ELO change (descending)
    upsets = upsets.sort_values('Elo_Change', ascending=False).reset_index(drop=True)
    
    # Create figure with table
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Apply custom color scheme
    config.apply_color_scheme(ax)
    
    # Prepare table data
    table_data = []
    for i, (_, row) in enumerate(upsets.iterrows()):
        # Format score as "Winner Score - Loser Score"
        if row['Outcome'] == 'W':
            score = f"{row['Points_For']}-{row['Points_Against']}"
        else:
            score = f"{row['Points_Against']}-{row['Points_For']}"
        
        table_data.append([
            f"{i+1}",
            f"{row['Team']}",
            f"{row['Opponent']}",
            f"{row['Date'].strftime('%Y-%m-%d')}",
            score,
            f"+{row['Elo_Change']:.0f}",
            f"{row['Outcome']}"
        ])
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=['Rank', 'Winner', 'Loser', 'Date', 'Score', 'ELO Change', 'Result'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)
    
    # Color the header row with custom colors
    for i in range(7):
        table[(0, i)].set_facecolor(config.COLORS['primary'])
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color alternating rows with custom colors
    for i in range(1, len(table_data) + 1):
        for j in range(7):
            if i % 2 == 0:
                table[(i, j)].set_facecolor(config.COLORS['background'])
            else:
                table[(i, j)].set_facecolor('#f8f9fa')  # Very light gray
    
    # Highlight the ELO change column (now column 5)
    for i in range(1, len(table_data) + 1):
        table[(i, 5)].set_facecolor('#e8f5e8')  # Light green
        table[(i, 5)].set_text_props(weight='bold', color=config.COLORS['success'])
    
    # Highlight the score column (column 4)
    for i in range(1, len(table_data) + 1):
        table[(i, 4)].set_facecolor('#fff3e0')  # Light orange
        table[(i, 4)].set_text_props(weight='bold', color=config.COLORS['warning'])
    
    # Set title with custom colors
    plt.title('Top 10 Biggest Upsets in NFL History', 
              fontsize=18, fontweight='bold', pad=20, color=config.COLORS['text_primary'])
    
    # Add subtitle with custom colors
    plt.figtext(0.5, 0.02, 'Based on ELO rating changes - higher values indicate bigger upsets', 
                ha='center', fontsize=10, style='italic', color=config.COLORS['text_secondary'])
    
    plt.tight_layout()
    
    # Save figure
    output_path = config.OUTPUT_DIR / 'biggest_swings.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=config.COLORS['background'])
    print(f"Saved top 10 upsets table to {output_path}")
    
    if config.SHOW_PLOTS:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory

@safefig
def plot_parity_metrics(parity_df: pd.DataFrame, config_dict: Dict = None) -> None:
    """
    Create a visualization of basic league parity metrics over time.
    
    This is a simplified parity metrics plot that works with the comprehensive
    parity data structure. For advanced parity analysis, use plot_combined_parity.
    
    Args:
        parity_df: DataFrame with parity metrics
        config_dict: Plot configuration dictionary
    """
    if not config.is_plot_enabled('parity_metrics'):
        print("Parity metrics plot disabled")
        return
    
    print("Creating basic parity metrics plot...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Apply color scheme to all subplots
    for ax in [ax1, ax2, ax3, ax4]:
        config.apply_color_scheme(ax)
    
    # ELO Standard Deviation over time (Dispersion_STD)
    if 'Dispersion_STD' in parity_df.columns:
        ax1.plot(parity_df['Season'], parity_df['Dispersion_STD'], linewidth=3, color=config.COLORS['primary'], alpha=0.8)
        ax1.set_xlabel('Season', fontweight='medium')
        ax1.set_ylabel('ELO Standard Deviation', fontweight='medium')
        ax1.set_title('League Parity: ELO Standard Deviation', fontweight='bold', fontsize=12)
    
    # ELO Range over time
    if 'ELO_Range' in parity_df.columns:
        ax2.plot(parity_df['Season'], parity_df['ELO_Range'], linewidth=3, color=config.COLORS['accent'], alpha=0.8)
        ax2.set_xlabel('Season', fontweight='medium')
        ax2.set_ylabel('ELO Range (Max - Min)', fontweight='medium')
        ax2.set_title('League Parity: ELO Range', fontweight='bold', fontsize=12)
    
    # Rank Stability over time
    if 'RankStability_Spearman' in parity_df.columns:
        ax3.plot(parity_df['Season'], parity_df['RankStability_Spearman'], linewidth=3, color=config.COLORS['success'], alpha=0.8)
        ax3.set_xlabel('Season', fontweight='medium')
        ax3.set_ylabel('Rank Stability (Spearman)', fontweight='medium')
        ax3.set_title('League Parity: Rank Stability', fontweight='bold', fontsize=12)
    
    # Upset Gap over time
    if 'Upset_Gap' in parity_df.columns:
        ax4.plot(parity_df['Season'], parity_df['Upset_Gap'], linewidth=3, color=config.COLORS['info'], alpha=0.8)
        ax4.set_xlabel('Season', fontweight='medium')
        ax4.set_ylabel('Upset Gap', fontweight='medium')
        ax4.set_title('League Parity: Upset Gap', fontweight='bold', fontsize=12)
    
    plt.suptitle('NFL League Parity Metrics Over Time (Basic)', fontsize=16, fontweight='bold', color=config.COLORS['text_primary'])
    plt.tight_layout()
    
    # Save figure
    output_path = config.OUTPUT_DIR / 'parity_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=config.COLORS['background'])
    print(f"Saved basic parity metrics to {output_path}")
    
    if config.SHOW_PLOTS:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory

@safefig
def plot_combined_parity(pm: pd.DataFrame, config_dict: Dict = None) -> None:
    """
    Create a combined parity plot showing both components and composite index.
    
    This combines the individual parity components and the composite parity index
    into a single comprehensive visualization.
    
    Args:
        pm: DataFrame with parity metrics including z-scores and composite index
        config_dict: Plot configuration dictionary
    """
    if not config.is_plot_enabled('combined_parity'):
        print("Combined parity plot disabled")
        return
    
    print("Creating combined parity plot...")
    
    cfg = config.get_plot_config('parity')
    pcfg = config.get_plot_config('parity_plots')
    
    # Check for required columns
    keep_cols = [c for c in ["Z_Dispersion", "Z_RankStab", "Z_UpsetGap", "Z_Entropy"] if c in pm.columns]
    if not keep_cols or 'Parity_Index' not in pm.columns:
        print("No parity component z-scores or composite index found.")
        return
    
    # Prepare data
    df = pm[["Season"] + keep_cols + ["Parity_Index"]].copy()
    
    # Apply smoothing
    k = max(1, int(cfg['smooth_ma']))
    for c in keep_cols + ["Parity_Index"]:
        df[c] = _moving_avg(df[c], k)
    
    # Rename for legend clarity
    rename_map = {
        "Z_Dispersion": "Dispersion (↓std = ↑parity)",
        "Z_RankStab": "Rank Stability (↓rho = ↑parity)",
        "Z_UpsetGap": "Upset Gap (↑ = ↑parity)",
        "Z_Entropy": "Forecast Entropy (↑ = ↑parity)",
    }
    labels = [rename_map.get(c, c) for c in keep_cols]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Apply color scheme to both subplots
    config.apply_color_scheme(ax1)
    config.apply_color_scheme(ax2)
    
    # Top plot: Parity components
    colors = config.COLORS['series'][:len(keep_cols)]
    for i, (c, lab) in enumerate(zip(keep_cols, labels)):
        ax1.plot(df["Season"], df[c], linewidth=2.0, alpha=0.9, 
                label=lab, color=colors[i % len(colors)])
    
    ax1.axhline(0, linewidth=2, alpha=0.6, color=config.COLORS['secondary'], linestyle='--')
    ax1.set_title("Parity Components (Z-scores; oriented so higher = more parity)", 
                 fontsize=14, fontweight='bold', color=config.COLORS['text_primary'])
    ax1.set_ylabel("Z-score", fontweight='medium')
    ax1.legend(ncol=pcfg['legend_cols'], fontsize=9, framealpha=0.9, edgecolor=config.COLORS['border'])
    
    # Bottom plot: Composite parity index
    ax2.plot(df["Season"], df["Parity_Index"], linewidth=3, color=config.COLORS['primary'], alpha=0.9)
    ax2.axhline(df["Parity_Index"].mean(), linewidth=2, alpha=0.6, 
               color=config.COLORS['accent'], linestyle='--', label=f'Mean: {df["Parity_Index"].mean():.2f}')
    ax2.set_title("Composite Parity Index (higher = more parity)", 
                 fontsize=14, fontweight='bold', color=config.COLORS['text_primary'])
    ax2.set_xlabel("Season", fontweight='medium')
    ax2.set_ylabel("Index (weighted z-score)", fontweight='medium')
    ax2.legend(fontsize=10, framealpha=0.9, edgecolor=config.COLORS['border'])
    
    # Overall title
    fig.suptitle("NFL League Parity Analysis", fontsize=16, fontweight='bold', y=0.98, color=config.COLORS['text_primary'])
    
    plt.tight_layout()
    
    # Save figure
    output_path = config.OUTPUT_DIR / 'combined_parity.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=config.COLORS['background'])
    print(f"Saved combined parity plot to {output_path}")
    
    if config.SHOW_PLOTS:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory

@safefig
def plot_season_delta_normalized(season_team_summary: pd.DataFrame, season: int = None, top_n: int = None) -> None:
    """
    Create a normalized ELO change graph showing team performance changes from start to end of season.
    
    Args:
        season_team_summary: DataFrame with season team summaries
        season: Season to plot (default: 2024)
        top_n: Number of teams to show (default: all)
    """
    if not config.is_plot_enabled('delta_normalized'):
        print("Delta normalized plot disabled")
        return
    
    print("Creating season delta normalized plot...")
    
    cfg = config.get_plot_config('delta_normalized')
    if season is None:
        season = cfg['season']
    if top_n is None:
        top_n = cfg['top_n']
    
    df = season_team_summary[season_team_summary["Season"] == season].copy()
    if df.empty:
        print(f"No data for season {season}")
        return
    
    df["Delta"] = df["Elo_Post"] - df["Elo_Pre"]
    df = df.sort_values("Delta", ascending=False)
    if top_n:
        df = df.head(top_n)
    df["y"] = np.arange(len(df))[::-1]
    
    fig = plt.figure(figsize=(12, 0.48 * len(df) + 2), constrained_layout=True)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.2, 6.0])
    ax_labels = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1], sharey=ax_labels)
    
    ax_labels.set_xlim(0, 1)
    ax_labels.set_ylim(-0.5, len(df) - 0.5)
    ax_labels.invert_yaxis()
    ax_labels.axis("off")
    for _, r in df.iterrows():
        ax_labels.text(0.98, r["y"], r["Team"], ha="right", va="center", fontsize=10)
    
    ax.set_ylim(-0.5, len(df) - 0.5)
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.axvline(0, linewidth=1, alpha=0.5)
    
    max_abs = float(np.nanmax(np.abs(df["Delta"]))) if len(df) else 0.0
    ax.set_xlim(-max_abs*1.1, max_abs*1.1)
    
    for _, r in df.iterrows():
        y = r["y"]
        d = float(r["Delta"])
        color = "#2ca02c" if d >= 0 else "#d62728"
        ax.plot([0, d], [y, y], color=color, linewidth=3.2, solid_capstyle="round")
        ax.scatter([d], [y], s=26, color=color, zorder=3)
        nudg = max_abs*0.02 if max_abs > 0 else 5.0
        ax.text(d + (nudg if d >= 0 else -nudg), y, f"{d:+.1f}",
                va="center", ha="left" if d >= 0 else "right", fontsize=9, color=color)
    
    ax.set_xlabel("Δ Elo (End − Start)")
    ax.set_title(f"Season {season} — Normalized Elo Change (0 → Δ)")
    
    # Save figure
    output_path = config.OUTPUT_DIR / f'delta_normalized_{season}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved delta normalized plot to {output_path}")
    
    if config.SHOW_PLOTS:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory

@safefig
def plot_conference_balance(season_conference_summary: pd.DataFrame) -> None:
    """
    Create a conference balance tracker showing NFC vs AFC ELO over time.
    
    Args:
        season_conference_summary: DataFrame with conference summaries
    """
    if not config.is_plot_enabled('conference_balance'):
        print("Conference balance plot disabled")
        return
    
    print("Creating conference balance plot...")
    
    cfg = config.get_plot_config('conference_balance')
    
    df = season_conference_summary.copy()
    p = df.pivot_table(index="Season", columns="Conference", values="EloMean", aggfunc="mean")
    p = p.dropna(subset=["AFC","NFC"], how="any")
    p["NFC_minus_AFC"] = p["NFC"] - p["AFC"]
    
    fig, ax = plt.subplots(figsize=cfg["figsize"])
    
    # Apply custom color scheme
    config.apply_color_scheme(ax)
    
    # Plot with blue color
    ax.plot(p.index, p["NFC_minus_AFC"], linewidth=3, color=config.COLORS['info'], alpha=0.9)
    ax.axhline(0, linewidth=2, alpha=0.6, color=config.COLORS['secondary'], linestyle='--')
    ax.set_title("Conference Balance Tracker (NFC − AFC Mean Elo)", 
                fontsize=14, fontweight='bold', color=config.COLORS['text_primary'])
    ax.set_xlabel("Season", fontweight='medium')
    ax.set_ylabel("Δ Elo", fontweight='medium')
    
    # Save figure
    output_path = config.OUTPUT_DIR / 'conference_balance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=config.COLORS['background'])
    print(f"Saved conference balance plot to {output_path}")
    
    plt.tight_layout()
    if config.SHOW_PLOTS:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory

@safefig
def plot_luck_index(luck_df: pd.DataFrame, season: int = None, top_n: int = None, use_abs: bool = False) -> None:
    """
    Create a lollipop plot of Luck Index for a given season.
    
    Args:
        luck_df: DataFrame with luck index data
        season: Season to plot (default: most recent)
        top_n: Number of teams to show (default: 20)
        use_abs: Whether to rank by absolute luck value
    """
    if not config.is_plot_enabled('luck_index'):
        print("Luck index plot disabled")
        return
    
    print("Creating luck index plot...")
    
    cfg = config.get_plot_config('luck_index')
    if season is None:
        season = int(luck_df["Season"].max())
    if top_n is None:
        top_n = cfg["top_n"]
    
    df = luck_df[luck_df["Season"] == season].copy()
    if df.empty:
        print(f"No luck data for season {season}.")
        return
    
    if use_abs:
        df = df.assign(_rank_key=df["Luck_Index"].abs())
    else:
        df = df.assign(_rank_key=df["Luck_Index"])
    
    df = df.sort_values("_rank_key", ascending=False)
    if top_n:
        df = df.head(int(top_n))
    df = df.assign(y=np.arange(len(df))[::-1])
    
    fig = plt.figure(figsize=(12, max(2, 0.5 * len(df) + 1)))
    ax = fig.add_subplot(111)
    
    ax.axvline(0, linewidth=1, alpha=0.5)
    for _, r in df.iterrows():
        val = float(r["Luck_Index"])
        color = "#2ca02c" if val >= 0 else "#d62728"
        ax.plot([0, val], [r["y"], r["y"]], linewidth=3.2, color=color, solid_capstyle="round")
        ax.scatter([val], [r["y"]], s=26, color=color, zorder=3)
        nudg = (df["_rank_key"].max() or 1) * 0.02
        ax.text(val + (nudg if val >= 0 else -nudg), r["y"], f"{val:+.2f}",
                va="center", ha="left" if val >= 0 else "right", fontsize=9, color=color)
    
    ax.set_yticks(df["y"])
    ax.set_yticklabels(df["Team"])
    ax.invert_yaxis()
    ax.set_xlabel("Luck Index (Actual Wins − Expected Wins)")
    ax.set_title(f"Luck Index — Season {season}" + (" (Top-N)" if top_n else ""))
    
    # Save figure
    output_path = config.OUTPUT_DIR / f'luck_index_{season}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved luck index plot to {output_path}")
    
    plt.tight_layout()
    if config.SHOW_PLOTS:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory

@safefig
def plot_calibration_and_brier(
    curve: pd.DataFrame,
    perf: pd.DataFrame,
    show_ci: bool | None = None
) -> None:
    """
    Plot reliability curve (probability calibration) and season-level Brier/LogLoss series.

    Note:
        Confidence intervals are omitted by default for clarity; can be added later if needed.
    """
    if not config.is_plot_enabled('calibration_plots'):
        print("Calibration plots disabled")
        return
    
    print("Creating reliability curve...")
    
    cfg = config.get_plot_config('calibration')
    if show_ci is None: show_ci = cfg.get('show_ci', False)

    # Calibration curve
    fig = plt.figure(figsize=(6.5, 6.0))
    ax = fig.add_subplot(111)
    
    # Apply custom color scheme
    config.apply_color_scheme(ax)
    
    # Perfect calibration line
    ax.plot([0,1],[0,1], linewidth=2, alpha=0.7, color=config.COLORS['secondary'], linestyle='--', label='Perfect Calibration')
    
    # Scatter plot with custom colors
    ax.scatter(curve["Pwin_mean"], curve["Outcome_mean"], s=60, 
               color=config.COLORS['accent'], alpha=0.8, edgecolors=config.COLORS['primary'], linewidth=1.5)
    
    ax.set_xlabel("Predicted Win Prob (mean in bin)", fontsize=12, fontweight='medium')
    ax.set_ylabel("Actual Outcome Rate (mean in bin)", fontsize=12, fontweight='medium')
    ax.set_title("Reliability Curve (A-side)", fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    ax.legend(loc='upper left', framealpha=0.9, edgecolor=config.COLORS['border'])
    
    # Save calibration curve
    output_path = config.OUTPUT_DIR / 'reliability_curve.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=config.COLORS['background'])
    print(f"Saved reliability curve to {output_path}")
    
    plt.tight_layout()
    if config.SHOW_PLOTS:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory

@safefig
def plot_murphy_diagram(
    sharpness_df: pd.DataFrame,
    brier_decomp_df: pd.DataFrame
) -> None:
    """
    Plot Murphy diagram: Sharpness vs Reliability.
    
    A Murphy diagram reveals whether the model is appropriately confident:
    - High sharpness + high reliability = overconfident (makes strong calls but wrong)
    - Low sharpness + high reliability = timid and poorly calibrated
    - High sharpness + low reliability = appropriately confident (strong calls, well-calibrated)
    - Low sharpness + low reliability = overly cautious (weak calls, well-calibrated)
    
    Args:
        sharpness_df: DataFrame with [Season, Sharpness, ...] from calculate_forecast_sharpness
        brier_decomp_df: DataFrame with [Season, Reliability, ...] from calculate_brier_decomposition
    """
    if not config.is_plot_enabled('calibration_plots'):
        print("Calibration plots disabled (Murphy diagram skipped)")
        return
    
    print("Creating Murphy diagram (Sharpness vs Reliability)...")
    
    # Merge sharpness and reliability data
    murphy_data = sharpness_df[['Season', 'Sharpness']].merge(
        brier_decomp_df[['Season', 'Reliability']],
        on='Season',
        how='inner'
    )
    
    if murphy_data.empty:
        print("No data available for Murphy diagram")
        return
    
    # Create figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    # Apply custom color scheme
    config.apply_color_scheme(ax)
    
    # Scatter plot: Sharpness (x-axis) vs Reliability (y-axis)
    # Lower reliability is better (well-calibrated)
    ax.scatter(murphy_data['Sharpness'], murphy_data['Reliability'],
               s=80, alpha=0.7, color=config.COLORS['accent'],
               edgecolors=config.COLORS['primary'], linewidth=1.5,
               zorder=3)
    
    # Add season labels for recent seasons
    recent_seasons = murphy_data.nlargest(5, 'Season')
    for _, row in recent_seasons.iterrows():
        ax.annotate(int(row['Season']),
                   (row['Sharpness'], row['Reliability']),
                   fontsize=8, alpha=0.7,
                   xytext=(5, 5), textcoords='offset points')
    
    # Add quadrant labels
    x_min, x_max = murphy_data['Sharpness'].min(), murphy_data['Sharpness'].max()
    y_min, y_max = murphy_data['Reliability'].min(), murphy_data['Reliability'].max()
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    
    # Quadrant interpretations
    ax.text(x_max * 0.95, y_max * 0.95, 'Overconfident\n(High Sharpness,\nHigh Reliability)',
            ha='right', va='top', fontsize=9, alpha=0.6,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax.text(x_min * 1.05, y_max * 0.95, 'Timid & Poorly Calibrated\n(Low Sharpness,\nHigh Reliability)',
            ha='left', va='top', fontsize=9, alpha=0.6,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax.text(x_max * 0.95, y_min * 1.05, 'Appropriately Confident\n(High Sharpness,\nLow Reliability)',
            ha='right', va='bottom', fontsize=9, alpha=0.6,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax.text(x_min * 1.05, y_min * 1.05, 'Overly Cautious\n(Low Sharpness,\nLow Reliability)',
            ha='left', va='bottom', fontsize=9, alpha=0.6,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    ax.set_xlabel("Sharpness (Std Dev of Predicted Probabilities)", 
                  fontsize=12, fontweight='medium')
    ax.set_ylabel("Reliability (Calibration Error)", 
                  fontsize=12, fontweight='medium')
    ax.set_title("Murphy Diagram: Sharpness vs Reliability", 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add grid for easier reading
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Save figure
    output_path = config.OUTPUT_DIR / 'murphy_diagram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor=config.COLORS['background'])
    print(f"Saved Murphy diagram to {output_path}")
    
    plt.tight_layout()
    if config.SHOW_PLOTS:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory

@safefig
def plot_brier_score_by_season(brier_decomp_df: pd.DataFrame) -> None:
    """
    Plot average Brier score by season over time.
    
    Args:
        brier_decomp_df: DataFrame with [Season, Brier, ...] from calculate_brier_decomposition
    """
    if not config.is_plot_enabled('calibration_plots'):
        print("Calibration plots disabled (Brier score by season skipped)")
        return
    
    print("Creating Brier score by season plot...")
    
    if brier_decomp_df.empty or 'Brier' not in brier_decomp_df.columns:
        print("No Brier score data available")
        return
    
    # Sort by season
    brier_data = brier_decomp_df[['Season', 'Brier']].copy().sort_values('Season')
    
    # Create figure
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    
    # Apply custom color scheme
    config.apply_color_scheme(ax)
    
    # Plot line with markers
    ax.plot(brier_data['Season'], brier_data['Brier'],
            marker='o', markersize=5, linewidth=2,
            color=config.COLORS['accent'], alpha=0.8,
            label='Brier Score', zorder=3)
    
    # Add horizontal line for average
    avg_brier = brier_data['Brier'].mean()
    ax.axhline(y=avg_brier, color=config.COLORS['secondary'], 
               linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'Average: {avg_brier:.4f}', zorder=1)
    
    # Add shaded region for typical range (0.18 to 0.22 as mentioned in paper)
    ax.axhspan(0.18, 0.22, alpha=0.15, color=config.COLORS['secondary'],
               label='Typical Range (0.18-0.22)', zorder=0)
    
    ax.set_xlabel("Season", fontsize=12, fontweight='medium')
    ax.set_ylabel("Brier Score", fontsize=12, fontweight='medium')
    ax.set_title("Average Brier Score by Season", 
                  fontsize=14, fontweight='bold', pad=20)
    
    # Format x-axis to show years clearly
    ax.set_xlim(brier_data['Season'].min() - 1, brier_data['Season'].max() + 1)
    ax.tick_params(axis='x', rotation=45)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
    
    # Add legend
    ax.legend(loc='best', framealpha=0.9, edgecolor=config.COLORS['border'])
    
    # Save figure
    output_path = config.OUTPUT_DIR / 'brier_score_by_season.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor=config.COLORS['background'])
    print(f"Saved Brier score by season plot to {output_path}")
    
    plt.tight_layout()
    if config.SHOW_PLOTS:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory

@safefig
def plot_logloss_by_season(calibration_perf: pd.DataFrame) -> None:
    """
    Plot average LogLoss by season over time.
    
    Args:
        calibration_perf: DataFrame with [Season, LogLoss, ...] from build_calibration_bins
    """
    if not config.is_plot_enabled('calibration_plots'):
        print("Calibration plots disabled (LogLoss by season skipped)")
        return
    
    print("Creating LogLoss by season plot...")
    
    if calibration_perf.empty or 'LogLoss' not in calibration_perf.columns:
        print("No LogLoss data available")
        return
    
    # Sort by season
    logloss_data = calibration_perf[['Season', 'LogLoss']].copy().sort_values('Season')
    
    # Create figure
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    
    # Apply custom color scheme
    config.apply_color_scheme(ax)
    
    # Plot line with markers
    ax.plot(logloss_data['Season'], logloss_data['LogLoss'],
            marker='o', markersize=5, linewidth=2,
            color=config.COLORS['accent'], alpha=0.8,
            label='LogLoss', zorder=3)
    
    # Add horizontal line for average
    avg_logloss = logloss_data['LogLoss'].mean()
    ax.axhline(y=avg_logloss, color=config.COLORS['secondary'], 
               linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'Average: {avg_logloss:.4f}', zorder=1)
    
    # Add shaded region for typical range (0.53 to 0.64 as mentioned in paper)
    ax.axhspan(0.53, 0.64, alpha=0.15, color=config.COLORS['secondary'],
               label='Typical Range (0.53-0.64)', zorder=0)
    
    ax.set_xlabel("Season", fontsize=12, fontweight='medium')
    ax.set_ylabel("LogLoss", fontsize=12, fontweight='medium')
    ax.set_title("Average LogLoss by Season", 
                  fontsize=14, fontweight='bold', pad=20)
    
    # Format x-axis to show years clearly
    ax.set_xlim(logloss_data['Season'].min() - 1, logloss_data['Season'].max() + 1)
    ax.tick_params(axis='x', rotation=45)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
    
    # Add legend
    ax.legend(loc='best', framealpha=0.9, edgecolor=config.COLORS['border'])
    
    # Save figure
    output_path = config.OUTPUT_DIR / 'logloss_by_season.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor=config.COLORS['background'])
    print(f"Saved LogLoss by season plot to {output_path}")
    
    plt.tight_layout()
    if config.SHOW_PLOTS:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory

@safefig
def plot_any_given_sunday_line(heat: pd.DataFrame) -> None:
    """
    Plot league-level 'Any Given Sunday' meter:
        Weighted average of (actual − expected) upset rates by season.
        > 0 : season had more upsets than Elo implied (more chaos).
        < 0 : fewer upsets than expected (chalky season).
    """
    if not config.is_plot_enabled('any_given_sunday'):
        print("Any Given Sunday meter disabled")
        return
    
    print("Creating Any Given Sunday meter...")
    
    if heat.empty:
        print("No upset gap data."); return
    agg = (
        heat.groupby("Season", as_index=False)
            .apply(lambda x: pd.Series({
                "Gap": np.average(x["Gap"], weights=x["N"])
            }))
            .reset_index(drop=True)
            .sort_values("Season")
    )
    fig, ax = plt.subplots(figsize=(10,4))
    
    # Apply custom color scheme
    config.apply_color_scheme(ax)
    
    # Main line plot
    ax.plot(agg["Season"], agg["Gap"], linewidth=3, color=config.COLORS['primary'], alpha=0.9)
    
    # Zero line
    ax.axhline(0, linewidth=2, alpha=0.6, color=config.COLORS['secondary'], linestyle='-')
    
    ax.set_title('"Any Given Sunday" Meter — Actual minus Expected Upsets', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Season", fontsize=12, fontweight='medium')
    ax.set_ylabel("Upset Gap", fontsize=12, fontweight='medium')
    
    # Save figure
    output_path = config.OUTPUT_DIR / 'any_given_sunday_meter.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=config.COLORS['background'])
    print(f"Saved Any Given Sunday meter to {output_path}")
    
    plt.tight_layout()
    if config.SHOW_PLOTS:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory

@safefig
def plot_luck_timelines(luck_df: pd.DataFrame, teams: list = None, use_rolling: bool = True, 
                       window: int = 3, figsize: tuple = (12, 7), line_alpha: float = 0.25,
                       team_linewidth: float = 0.8, league_linewidth: float = 2.5) -> None:
    """
    Plot each team's Luck Index (Actual Wins − Expected Wins) over time.
    
    This mirrors the 'league tapestry' Elo chart, but visualizes whether teams
    consistently over- or under-performed model expectations.
    
    Args:
        luck_df: DataFrame with luck index data
        teams: List of team codes to plot; if None → all franchises
        use_rolling: Whether to apply rolling mean smoothing across seasons
        window: Rolling window length (seasons) for smoothing
        figsize: Figure size in inches
        line_alpha: Line transparency for individual teams
        team_linewidth: Line width for individual team traces
        league_linewidth: Line width for league-average trace
    """
    if not config.is_plot_enabled('luck_timelines'):
        print("Luck timelines plot disabled")
        return
    
    print("Creating luck timelines plot...")
    
    cfg = config.get_plot_config('luck_timelines')
    if teams is None:
        teams = cfg.get('teams', None)
    if use_rolling is None:
        use_rolling = cfg.get('use_rolling', True)
    if window is None:
        window = cfg.get('window', 3)
    if figsize is None:
        figsize = cfg.get('figsize', (12, 7))
    
    df = luck_df.copy()
    df = df.dropna(subset=["Season","Team","Luck_Index"])
    
    # Compute league-average luck FIRST (before rolling) - should be exactly 0
    # Calculate as (Total Actual Wins - Total Expected Wins) / Total Games
    # This should always be 0 because sum of actual wins = sum of expected wins = number of games
    league_raw = (
        luck_df.groupby("Season", as_index=False)
        .agg({
            'ActualWins': 'sum',
            'ExpWins': 'sum',
            'Games': 'sum'
        })
    )
    # Calculate league luck as (total actual - total expected) / total games
    # This should always be 0 (or very close due to rounding)
    league_raw['LeagueMean'] = (league_raw['ActualWins'] - league_raw['ExpWins']) / league_raw['Games']
    
    # Rolling smooth per team if desired
    if use_rolling:
        df = (
            df.sort_values(["Team","Season"])
              .groupby("Team", group_keys=False)
              .apply(lambda x: x.assign(
                  Luck_Roll=lambda d: d["Luck_Index"].rolling(window, min_periods=1).mean()
              ))
        )
        ycol = "Luck_Roll"
        # Apply rolling to league average too
        league_raw['LeagueMean'] = league_raw['LeagueMean'].rolling(window=window, min_periods=1).mean()
    else:
        ycol = "Luck_Index"
    
    # Restrict teams if provided
    if teams is None:
        teams = sorted(df["Team"].unique().tolist())
    
    # Keep only Season and LeagueMean for plotting
    league = league_raw[['Season', 'LeagueMean']]
    
    plt.figure(figsize=figsize)
    for t in teams:
        g = df[df["Team"] == t].sort_values("Season")
        if not g.empty:
            plt.plot(g["Season"], g[ycol], linewidth=team_linewidth, alpha=line_alpha)
    
    # League-average trace and baseline at 0
    plt.plot(league["Season"], league["LeagueMean"],
             linewidth=league_linewidth, color="black", alpha=0.9, label="League Avg")
    plt.axhline(0, color="gray", linewidth=1.2, linestyle="--", alpha=0.8)
    
    plt.title(f"Luck Index Tapestry ({'Rolling ' + str(window) + ' Seasons' if use_rolling else 'Raw Season Values'})")
    plt.xlabel("Season")
    plt.ylabel("Luck Index (Actual Wins − Expected Wins)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    output_path = config.OUTPUT_DIR / 'luck_timelines.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved luck timelines to {output_path}")
    
    plt.tight_layout()
    if config.SHOW_PLOTS:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory

@safefig
def plot_sos_vs_elo(sos_df: pd.DataFrame, season: int = None, show_quadrants: bool = True) -> None:
    """
    Scatter plot of SoS (x) vs End-of-Season Elo (y) for a selected regular season.
    Quadrants highlight: tough schedule + strong finish, etc.
    
    Args:
        sos_df: DataFrame with SoS data
        season: Season to plot (default: most recent)
        show_quadrants: Whether to show quadrant lines
    """
    if not config.is_plot_enabled('sos_vs_elo'):
        print("SoS vs ELO plot disabled")
        return
    
    print("Creating SoS vs Final ELO plot...")
    
    cfg = config.get_plot_config('sos')
    if season is None:
        season = cfg.get('season')
        if season is None:
            season = int(sos_df["Season"].max())
    if show_quadrants is None:
        show_quadrants = cfg.get('show_quadrants', True)
    
    # Choose which SoS metric to use (default: SoS_PreMean)
    sos_metric = cfg.get('sos_metric', 'SoS_PreMean')
    sos_metric_options = ['SoS_PreMean', 'SoS_EndMean', 'SoS_PreWeighted', 'SoS_EndWeighted']
    if sos_metric not in sos_metric_options:
        sos_metric = 'SoS_PreMean'  # Fallback to default
    
    df = sos_df[sos_df["Season"] == int(season)].dropna(subset=[sos_metric, "Elo_Post"]).copy()
    if df.empty:
        print(f"No SoS data for season {season}.")
        return
    
    xmid = float(df[sos_metric].mean())
    ymid = float(df["Elo_Post"].mean())
    
    fig, ax = plt.subplots(figsize=(10,6))
    
    # Apply custom color scheme
    config.apply_color_scheme(ax)
    
    # Create scatter plot with team colors
    colors = [config.get_team_color(team) for team in df["Team"]]
    ax.scatter(df[sos_metric], df["Elo_Post"], s=60, alpha=0.8, c=colors, edgecolors=config.COLORS['primary'], linewidth=1.0)
    
    # Add team labels
    for _, r in df.iterrows():
        ax.text(r[sos_metric]+0.6, r["Elo_Post"]+0.6, r["Team"], fontsize=8, 
                color=config.COLORS['text_secondary'], fontweight='medium')
    
    if show_quadrants:
        ax.axvline(xmid, linewidth=2, alpha=0.6, color=config.COLORS['secondary'], linestyle='--')
        ax.axhline(ymid, linewidth=2, alpha=0.6, color=config.COLORS['secondary'], linestyle='--')
    
    sos_label_map = {
        'SoS_PreMean': 'SoS (Pre-Game Mean)',
        'SoS_EndMean': 'SoS (End-of-Season Mean)',
        'SoS_PreWeighted': 'SoS (Pre-Game, Weighted)',
        'SoS_EndWeighted': 'SoS (End-of-Season, Weighted)'
    }
    ax.set_xlabel(f"Strength of Schedule - {sos_label_map.get(sos_metric, sos_metric)}", fontweight='medium')
    ax.set_ylabel("End-of-Season Elo", fontweight='medium')
    ax.set_title(f"SoS vs Final Elo — Regular Season {season}", 
                fontsize=14, fontweight='bold', color=config.COLORS['text_primary'])
    
    # Save figure
    output_path = config.OUTPUT_DIR / f'sos_vs_elo_{season}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=config.COLORS['background'])
    print(f"Saved SoS vs ELO plot to {output_path}")
    
    plt.tight_layout()
    if config.SHOW_PLOTS:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory

@safefig
def export_team_story_card(season_team_summary: pd.DataFrame, luck_df: pd.DataFrame, 
                          timeline: pd.DataFrame, season: int = None, team: str = None, 
                          export_path: str = None) -> None:
    """
    One-page story card summarizing a team-season.
    Includes:
      • Start/End Elo, record in close games
      • Biggest single-game Elo swing (opponent/date/score)
      • Luck Index and SoS
      • Mini sparkline (weekly Elo)
    
    Args:
        season_team_summary: DataFrame with season team summaries
        luck_df: DataFrame with luck index data
        timeline: DataFrame with team timeline data
        season: Season to analyze (default: most recent)
        team: Team to analyze (default: first team in season)
        export_path: Path to save the PNG file
    """
    if not config.is_plot_enabled('story_card'):
        print("Team story card disabled")
        return
    
    print("Creating team story card...")
    
    cfg = config.get_plot_config('story_card')
    if season is None:
        season = cfg.get('season', int(season_team_summary["Season"].max()))
    if team is None:
        team = cfg.get('team', season_team_summary[season_team_summary["Season"]==int(season)]["Team"].iloc[0])
    if export_path is None:
        export_path = cfg.get('export_path', 'team_story_card.png')
    
    # Core stats
    st = season_team_summary[(season_team_summary["Season"]==int(season)) & (season_team_summary["Team"]==team)]
    if st.empty:
        print(f"No summary for {team} {season}.")
        return
    
    elo_start = float(st["Elo_Pre"].iloc[0])
    elo_end = float(st["Elo_Post"].iloc[0])
    
    # Luck index
    lk = luck_df[(luck_df["Season"]==int(season)) & (luck_df["Team"]==team)]
    luck_idx = float(lk["Luck_Index"].iloc[0]) if not lk.empty else np.nan
    
    # Biggest swing from timeline
    team_timeline = timeline[(timeline["Season"]==int(season)) & (timeline["Team"]==team)]
    if not team_timeline.empty:
        biggest_swing_idx = team_timeline["Elo_Change"].abs().idxmax()
        biggest_swing = team_timeline.loc[biggest_swing_idx]
        big_swing = float(biggest_swing["Elo_Change"])
        sw_date = pd.to_datetime(biggest_swing["Date"]).date()
        sw_vs = biggest_swing["Opponent"]
        sw_score = f"{biggest_swing['Points_For']}-{biggest_swing['Points_Against']}"
    else:
        big_swing, sw_date, sw_vs, sw_score = (np.nan, "", "", "")
    
    # Layout
    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[1,1.2], width_ratios=[1.2,1,1])
    
    # Title
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(0.02, 0.65, f"{team} — {season}", fontsize=20, weight="bold")
    ax_title.text(0.02, 0.35, f"Team Story Card", fontsize=12, alpha=0.7)
    
    # KPIs block
    ax_kpi = fig.add_subplot(gs[1,0])
    ax_kpi.axis("off")
    lines = [
        (f"Elo Start", f"{elo_start:.1f}"),
        (f"Elo End", f"{elo_end:.1f}"),
        ("Luck Index", f"{luck_idx:+.2f}" if not np.isnan(luck_idx) else "—"),
        ("Games Played", f"{len(team_timeline)}" if not team_timeline.empty else "—"),
    ]
    y = 0.95
    for k,v in lines:
        ax_kpi.text(0.02, y, k, fontsize=12, alpha=0.8)
        ax_kpi.text(0.75, y, v, fontsize=12, weight="bold", ha="right")
        y -= 0.16
    
    # Biggest swing block
    ax_sw = fig.add_subplot(gs[1,1])
    ax_sw.axis("off")
    ax_sw.text(0.02, 0.9, "Biggest Swing", fontsize=12, weight="bold")
    if not np.isnan(big_swing):
        ax_sw.text(0.02, 0.65, f"{big_swing:.1f} Elo vs {sw_vs}", fontsize=12, weight="bold")
        ax_sw.text(0.02, 0.40, f"{sw_date} ({sw_score})", fontsize=11, alpha=0.8)
    else:
        ax_sw.text(0.02, 0.65, "—", fontsize=12)
    
    # Sparkline
    ax_sp = fig.add_subplot(gs[1,2])
    if not team_timeline.empty:
        # Create a simple ELO progression - use the correct column name
        team_timeline_sorted = team_timeline.sort_values('Date')
        # Use the correct column name from timeline: 'Elo_Post'
        elo_col = 'Elo_Post'
        
        ax_sp.plot(range(len(team_timeline_sorted)), team_timeline_sorted[elo_col], lw=2, color='blue')
        ax_sp.set_title("Season ELO Progression", fontsize=11)
        ax_sp.set_xticks([])
        ax_sp.set_yticks([])
        ax_sp.spines[['top','right','left','bottom']].set_visible(False)
    else:
        ax_sp.axis("off")
        ax_sp.text(0.5,0.5,"No data", ha="center", va="center")
    
    fig.suptitle("Team Story Card", fontsize=12, y=0.99)
    
    # Save figure
    output_path = config.OUTPUT_DIR / export_path
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    print(f"Saved team story card to {output_path}")
    
    if config.SHOW_PLOTS:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function - this is where everything happens!
    
    This function orchestrates the entire analysis pipeline:
    1. Load the game data
    2. Calculate ELO ratings for all teams
    3. Create timelines and summaries
    4. Calculate advanced metrics (luck, parity, calibration, etc.)
    5. Generate visualizations
    6. Export results
    
    You can customize what runs by modifying the config object.
    """
    
    print("=" * 70)
    print(" " * 20 + "NFL ELO ANALYSIS TOOL")
    print("=" * 70)
    
    # Step 1: Load and process the raw data
    try:
        print("\n" + "-" * 70)
        print("STEP 1: Loading and Processing Data")
        print("-" * 70)
        raw_game_data = load_nfl_data()
        elo_games = calculate_elo_ratings(raw_game_data)
        team_timeline = create_team_timeline(elo_games)
        monthly_elo_data = create_monthly_elo_data(team_timeline)
        
        print("\n" + " " * 4 + "Data Summary")
        print(" " * 4 + "-" * 66)
        print(f"  Total games processed:     {len(elo_games):>10,}")
        print(f"  Unique teams:              {team_timeline['Team'].nunique():>10}")
        print(f"  Date range:                {str(team_timeline['Date'].min().date()):>10} to {team_timeline['Date'].max().date()}")
        print(f"  Seasons covered:           {team_timeline['Season'].min():>10} to {team_timeline['Season'].max()}")
        print(f"  Monthly data points:       {len(monthly_elo_data):>10,}")
        
    except FileNotFoundError as error:
        print(f"\n[ERROR] {error}")
        print("        Please ensure 'NFLELO_data.xlsx' exists in the current directory.")
        return
    except Exception as error:
        print(f"\n[ERROR] Processing data failed: {error}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Show what plots will be generated
    print("\n" + "-" * 70)
    print("STEP 2: Plot Configuration")
    print("-" * 70)
    enabled_plots = [name for name, enabled in config.PLOTS_ENABLED.items() if enabled]
    disabled_plots = [name for name, enabled in config.PLOTS_ENABLED.items() if not enabled]
    
    if enabled_plots:
        enabled_list = ', '.join(enabled_plots[:5])
        if len(enabled_plots) > 5:
            enabled_list += f" and {len(enabled_plots)-5} more"
        print(f"  Enabled plots ({len(enabled_plots)}):  {enabled_list}")
    if disabled_plots:
        disabled_list = ', '.join(disabled_plots[:5])
        if len(disabled_plots) > 5:
            disabled_list += f" and {len(disabled_plots)-5} more"
        print(f"  Disabled plots ({len(disabled_plots)}): {disabled_list}")
    
    print("\n  Configuration options:")
    print("    config.toggle_plot('plot_name')")
    print("    config.disable_all_plots()")
    print("    config.enable_all_plots()")
    
    # Step 3: Calculate summary statistics
    print("\n" + "-" * 70)
    print("STEP 3: Generating Summary Statistics")
    print("-" * 70)
    team_summary_statistics = calculate_team_summary_stats(team_timeline)
    biggest_elo_swings = find_biggest_swings(team_timeline, top_n=20)
    
    # Use the original elo_games data structure for calibration and upset analysis
    # (already created in calculate_elo_ratings)
    
    # Step 4: Create comprehensive analysis metrics
    print("\n" + "-" * 70)
    print("STEP 4: Calculating Advanced Metrics")
    print("-" * 70)
    season_team_summary = create_season_team_summary(team_timeline)
    season_conference_summary = create_season_conference_summary(season_team_summary)
    luck_index_data = build_luck_index(team_timeline, elo_games)
    average_luck_by_team = calculate_average_luck_by_team(luck_index_data)
    
    # Save luck index data to CSV files
    luck_index_csv_path = config.OUTPUT_DIR / 'luck_index_full.csv'
    luck_index_data.to_csv(luck_index_csv_path, index=False)
    print(f"  [OK] Saved luck index data: {luck_index_csv_path.name}")
    
    avg_luck_csv_path = config.OUTPUT_DIR / 'average_luck_by_team.csv'
    average_luck_by_team.to_csv(avg_luck_csv_path, index=False)
    print(f"  [OK] Saved average luck scores: {avg_luck_csv_path.name}")
    
    # Calculate calibration and forecast quality metrics
    calibration_curve, calibration_performance = build_calibration_bins(elo_games)
    brier_decomposition = calculate_brier_decomposition(elo_games)
    upset_map = build_upset_map(elo_games, elo_bin_edges=list(range(0, 401, 25)), min_bin_n=30)
    
    # Calculate strength of schedule and parity metrics
    strength_of_schedule_data = build_sos(season_team_summary, team_timeline, elo_games, weighting="home_adj")
    parity_metrics = build_parity_metrics_clean(season_team_summary, elo_games)
    
    # Calculate additional team metrics
    season_team_summary = calculate_volatility_and_dynasty(team_timeline, season_team_summary)
    forecast_sharpness_metrics = calculate_forecast_sharpness(elo_games)
    division_summary = create_division_summary(season_team_summary)
    
    # Step 5: Generate visualizations
    print("\n" + "-" * 70)
    print("STEP 5: Generating Visualizations")
    print("-" * 70)
    
    # Core visualizations - the main plots everyone wants to see
    plot_league_tapestry(monthly_elo_data)
    plot_team_small_multiples(monthly_elo_data)
    plot_season_ladder(team_timeline)
    
    # Team performance analysis plots
    plot_team_summary_stats(team_summary_statistics)
    plot_biggest_swings(biggest_elo_swings)
    
    # League-wide analysis plots
    plot_parity_metrics(parity_metrics)
    plot_combined_parity(parity_metrics)
    plot_season_delta_normalized(season_team_summary)
    plot_conference_balance(season_conference_summary)
    
    # Luck and calibration analysis plots
    plot_luck_index(luck_index_data)
    plot_calibration_and_brier(calibration_curve, calibration_performance)
    plot_murphy_diagram(forecast_sharpness_metrics, brier_decomposition)
    plot_brier_score_by_season(brier_decomposition)
    plot_logloss_by_season(calibration_performance)
    plot_any_given_sunday_line(upset_map)
    plot_luck_timelines(luck_index_data)
    
    # Strength of schedule and team story visualizations
    plot_sos_vs_elo(strength_of_schedule_data)
    export_team_story_card(season_team_summary, luck_index_data, team_timeline)
    
    # Step 6: Run sensitivity analysis (tests different parameter combinations)
    print("\n" + "=" * 70)
    print("STEP 6: Sensitivity Analysis")
    print("=" * 70)
    print("  Testing different ELO parameter combinations to find optimal settings...")
    sensitivity_results = run_sensitivity_analysis(raw_game_data)
    plot_sensitivity_analysis_table(sensitivity_results)
    
    # Step 7: Export comprehensive data
    print("\n" + "-" * 70)
    print("STEP 7: Exporting Comprehensive Data")
    print("-" * 70)
    comprehensive_data = export_comprehensive_data_csv(
        team_timeline, elo_games, season_team_summary, luck_index_data, strength_of_schedule_data
    )
    
    # Add more plot calls here as needed
    # plot_delta_normalized(timeline)
    # plot_division_heatmap(timeline)
    # plot_conference_balance(timeline)
    # etc.
    
    # Step 8: Display key findings
    print("\n" + "=" * 70)
    print("STEP 8: Key Findings")
    print("=" * 70)
    
    print("\n" + " " * 4 + "Current Top 5 Teams by ELO Rating")
    print(" " * 4 + "-" * 66)
    for rank, (_, team_row) in enumerate(team_summary_statistics.head(5).iterrows(), 1):
        print(f"  {rank:2d}. {team_row['Team']:3s}  {team_row['Current_ELO']:>7.0f} ELO")
    
    print("\n" + " " * 4 + "Biggest ELO Swing in History")
    print(" " * 4 + "-" * 66)
    if not biggest_elo_swings.empty:
        biggest_swing = biggest_elo_swings.iloc[0]
        print(f"  {biggest_swing['Team']} vs {biggest_swing['Opponent']}")
        print(f"  Date: {biggest_swing['Date'].strftime('%B %d, %Y')}")
        print(f"  ELO Change: {biggest_swing['Elo_Change']:+.0f} points")
    
    print("\n" + " " * 4 + "League Parity (Most Recent Season)")
    print(" " * 4 + "-" * 66)
    if not parity_metrics.empty:
        most_recent_season = parity_metrics.iloc[-1]
        if 'Dispersion_STD' in most_recent_season and 'ELO_Range' in most_recent_season:
            print(f"  Season: {most_recent_season['Season']}")
            print(f"  ELO Standard Deviation: {most_recent_season['Dispersion_STD']:>6.1f}")
            print(f"  ELO Range:              {most_recent_season['ELO_Range']:>6.0f} points")
        elif 'Parity_Index' in most_recent_season:
            print(f"  Season: {most_recent_season['Season']}")
            print(f"  Parity Index: {most_recent_season['Parity_Index']:>6.2f}")
    
    print("\n" + " " * 4 + "Luck Analysis (Average Across All Seasons)")
    print(" " * 4 + "-" * 66)
    print("  Most Lucky Teams (Top 5):")
    for rank, (_, team_row) in enumerate(average_luck_by_team.head(5).iterrows(), 1):
        print(f"    {rank:2d}. {team_row['Team']:3s}  {team_row['Avg_Luck_Index']:>+6.2f} per season "
              f"({team_row['Seasons']:.0f} seasons, {team_row['Overall_Luck_Index']:>+6.1f} total)")
    print("  Least Lucky Teams (Bottom 5):")
    for rank, (_, team_row) in enumerate(average_luck_by_team.tail(5).iterrows(), 1):
        print(f"    {rank:2d}. {team_row['Team']:3s}  {team_row['Avg_Luck_Index']:>+6.2f} per season "
              f"({team_row['Seasons']:.0f} seasons, {team_row['Overall_Luck_Index']:>+6.1f} total)")
    
    print("\n" + "=" * 70)
    print(" " * 25 + "ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"  Output directory: {config.OUTPUT_DIR.absolute()}")
    print("=" * 70)

if __name__ == "__main__":
    main()
