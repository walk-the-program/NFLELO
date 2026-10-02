# Game Impact Analysis - How a Single Game Affects All Metrics

This document demonstrates how a single NFL game impacts all calculations and metrics in the NFL ELO Analysis Tool. We'll walk through a hypothetical game step-by-step to show how every aspect of the system is updated.

---

## Hypothetical Game Scenario

**Game Details:**
- **Date**: October 15, 2023
- **Season**: 2023
- **Team A (Home)**: Kansas City Chiefs (KAN)
- **Team B (Away)**: Buffalo Bills (BUF)
- **Pre-Game ELO Ratings**: 
  - KAN: 1650 (elite team)
  - BUF: 1580 (very good team)
- **Final Score**: KAN 31, BUF 24
- **Result**: KAN wins by 7 points

**Context:**
- This is game #8 for both teams in the 2023 season
- Both teams have played 7 games prior to this matchup
- KAN is playing at home (Arrowhead Stadium)

---

## Step 1: Pre-Game Setup

### Initial State

**Kansas City Chiefs (KAN):**
- Current ELO: 1650
- Season record: 5-2 (5 wins, 2 losses)
- Season ELO change so far: +50 (started at 1600)
- Points For (season): 210
- Points Against (season): 175

**Buffalo Bills (BUF):**
- Current ELO: 1580
- Season record: 4-3 (4 wins, 3 losses)
- Season ELO change so far: -20 (started at 1600)
- Points For (season): 195
- Points Against (season): 185

---

## Step 2: ELO Rating Calculation

### 2.1 Apply Home Field Advantage

```
Ra = 1650 (KAN pre-game ELO)
Rb = 1580 (BUF pre-game ELO)
HFA = 55.0 (configurable home field advantage)

Ra_eff = Ra + HFA = 1650 + 55 = 1705
Rb_eff = Rb + 0 = 1580
```

**Effective Ratings:**
- KAN (home): 1705
- BUF (away): 1580
- **Difference**: 125 points (KAN favored)

### 2.2 Calculate Expected Win Probabilities

```
Ea = 1 / (1 + 10^((Rb_eff - Ra_eff) / 400))
Ea = 1 / (1 + 10^((1580 - 1705) / 400))
Ea = 1 / (1 + 10^(-125 / 400))
Ea = 1 / (1 + 10^(-0.3125))
Ea = 1 / (1 + 0.485)
Ea = 1 / 1.485
Ea = 0.673 (67.3% chance KAN wins)

Eb = 1 - Ea = 1 - 0.673 = 0.327 (32.7% chance BUF wins)
```

**Expected Outcomes:**
- KAN expected win probability: **67.3%**
- BUF expected win probability: **32.7%**

### 2.3 Determine Actual Outcome

```
Final Score: KAN 31, BUF 24
Result: KAN wins

Sa = 1.0 (KAN wins)
Sb = 0.0 (BUF loses)
PointDiff = 31 - 24 = 7 points
```

**Actual Outcomes:**
- KAN: 1.0 (win)
- BUF: 0.0 (loss)
- Point differential: 7 points (close game)

### 2.4 Calculate Margin of Victory Factor

**Configuration:**
- `USE_MARGIN_OF_VICTORY = True`
- `MOV_SCALING_METHOD = "elo_standard"`

```
margin = abs(PointDiff) = abs(7) = 7
elo_diff = abs(Ra_eff - Rb_eff) = abs(1705 - 1580) = 125

MOV_Factor = ln(margin + 1) * (2.2 / ((elo_diff * 0.001) + 2.2))
MOV_Factor = ln(8) * (2.2 / ((125 * 0.001) + 2.2))
MOV_Factor = 2.079 * (2.2 / (0.125 + 2.2))
MOV_Factor = 2.079 * (2.2 / 2.325)
MOV_Factor = 2.079 * 0.947
MOV_Factor = 1.97
MOV_Factor = min(1.97, 2.0) = 1.97
```

**Why is the MOV Factor So High (1.97) for a 7-Point Game?**

This seems counterintuitive at first! A 7-point win is only one score, yet it gets nearly the maximum MOV factor (2.0). Here's why:

**The Formula Has Two Parts:**

1. **`ln(margin + 1)` = ln(8) = 2.079**
   - The natural logarithm grows slowly, but even for small margins it's significant
   - ln(4) = 1.39 (3-point game)
   - ln(8) = 2.08 (7-point game) ← **We're here**
   - ln(15) = 2.71 (14-point game)
   - ln(21) = 3.05 (20-point game)
   - **The log function means a 7-point margin already gets a high base value**

2. **`(2.2 / ((elo_diff * 0.001) + 2.2))` = 0.947**
   - Since the teams are relatively close (125 point difference), this adjustment is near 1.0
   - If teams were far apart (300+ points), this would be much lower (~0.7)
   - **Because teams are close, the full logarithmic impact applies**

**The Result:**
- 2.079 × 0.947 = **1.97** (nearly the 2.0 cap)

**Why This Makes Sense:**
The MOV system is designed to recognize that:
- **Any margin of victory** (even 7 points) contains information about relative team strength
- When teams are **close in ELO** (125 points), even a small margin suggests the teams might be closer than the pre-game ratings indicated
- The **logarithmic scaling** prevents extreme blowouts from being 10x more impactful, but it still amplifies even moderate margins
- The **ELO difference adjustment** reduces impact only when there's a big mismatch (expected results)

**Comparison:**
- **3-point win** (very close): ln(4) × 0.947 = **1.32** (lower)
- **7-point win** (one score): ln(8) × 0.947 = **1.97** (high) ← **Our case**
- **14-point win** (two scores): ln(15) × 0.947 = **2.57** → capped at **2.0** (maximum)
- **21-point win** (blowout): ln(22) × 0.947 = **3.09** → capped at **2.0** (maximum)

**Key Insight:** The formula is designed so that even moderate margins (7-14 points) get high multipliers, especially when teams are close. The 2.0 cap prevents extreme blowouts from being too impactful, but it means that a 7-point win between close teams gets nearly the same multiplier as a 28-point blowout!

### 2.5 Calculate ELO Changes

**Configuration:**
- `K_FACTOR = 20.0`

```
Delta_A = K * MOV_Factor * (Sa - Ea)
Delta_A = 20 * 1.97 * (1.0 - 0.673)
Delta_A = 20 * 1.97 * 0.327
Delta_A = 12.9

Delta_B = K * MOV_Factor * (Sb - Eb)
Delta_B = 20 * 1.97 * (0.0 - 0.327)
Delta_B = 20 * 1.97 * (-0.327)
Delta_B = -12.9
```

**ELO Changes:**
- KAN: **+12.9 points**
- BUF: **-12.9 points**

### 2.6 Update ELO Ratings

```
Ra_new = Ra + Delta_A = 1650 + 12.9 = 1662.9
Rb_new = Rb + Delta_B = 1580 + (-12.9) = 1567.1
```

**New ELO Ratings:**
- KAN: **1662.9** (up from 1650)
- BUF: **1567.1** (down from 1580)

**Why KAN gained more than expected:**
- KAN was favored (67.3% win probability)
- But the game was close (7 points), so MOV factor was high (1.97)
- This amplifies the rating change even though it was an expected win
- **Without MOV**: KAN would gain only 6.5 points (20 * 0.327)
- **With MOV**: KAN gains 12.9 points (almost double!)

---

## Step 3: Update Game-Level Data Structures

### 3.1 ELO Games Record

A new record is added to `elo_games`:

```python
{
    'Date': '2023-10-15',
    'Season': 2023,
    'GameID': 12345,
    'TeamA': 'KAN',
    'TeamB': 'BUF',
    'A_Pre': 1650.0,
    'B_Pre': 1580.0,
    'A_Exp': 0.673,
    'B_Exp': 0.327,
    'A_Act': 1.0,
    'B_Act': 0.0,
    'A_Delta': 12.9,
    'B_Delta': -12.9,
    'A_Post': 1662.9,
    'B_Post': 1567.1,
    'A_Home': 1,
    'B_Home': 0,
    'A_Points': 31,
    'B_Points': 24,
    'PointDiff': 7,
    'MOV_Factor': 1.97,
    'Outcome': 'W'
}
```

### 3.2 Timeline Records

Two records are added to `timeline` (one per team):

**KAN Timeline Record:**
```python
{
    'Date': '2023-10-15',
    'Season': 2023,
    'Team': 'KAN',
    'Opponent': 'BUF',
    'Elo_Pre': 1650.0,
    'Elo_Post': 1662.9,
    'Elo_Change': 12.9,
    'Expected': 0.673,
    'Actual': 1.0,
    'Home': 1,
    'Points_For': 31,
    'Points_Against': 24,
    'Outcome': 'W'
}
```

**BUF Timeline Record:**
```python
{
    'Date': '2023-10-15',
    'Season': 2023,
    'Team': 'BUF',
    'Opponent': 'KAN',
    'Elo_Pre': 1580.0,
    'Elo_Post': 1567.1,
    'Elo_Change': -12.9,
    'Expected': 0.327,
    'Actual': 0.0,
    'Home': 0,
    'Points_For': 24,
    'Points_Against': 31,
    'Outcome': 'L'
}
```

---

## Step 4: Update Season-Level Metrics

### 4.1 Luck Index Updates

**For KAN:**
```
Before this game:
  ActualWins = 5
  ExpWins = sum of expected win probabilities from previous 7 games
  Let's say ExpWins = 4.8 (from previous games)

After this game:
  ActualWins = 5 + 1 = 6
  ExpWins = 4.8 + 0.673 = 5.473
  Luck_Index = 6 - 5.473 = +0.527

Close Game Check:
  PointDiff = 7 ≤ 7 (CLOSE_GAME_POINT_DIFF) ✓
  Expected = 0.673 (not in [0.35, 0.65] range)
  Is close game: YES (by point differential)

Close_ActualWins = previous close wins + 1 (if this was a close win)
Close_ExpWins = previous close expected + 0.673
Close_Luck_Index = Close_ActualWins - Close_ExpWins
```

**For BUF:**
```
Before this game:
  ActualWins = 4
  ExpWins = 4.2 (from previous games)

After this game:
  ActualWins = 4 + 0 = 4
  ExpWins = 4.2 + 0.327 = 4.527
  Luck_Index = 4 - 4.527 = -0.527

Close Game Check:
  PointDiff = 7 ≤ 7 ✓
  Is close game: YES

Close_ActualWins = previous close wins + 0 (lost)
Close_ExpWins = previous close expected + 0.327
Close_Luck_Index = Close_ActualWins - Close_ExpWins
```

**Interpretation:**
- KAN got slightly "lucky" (+0.527 wins above expectation)
- BUF got slightly "unlucky" (-0.527 wins below expectation)
- This is a close game, so the luck is in coin-flip situations

### 4.2 Pythagorean Expectation Updates

**For KAN:**
```
Before this game:
  Points_For = 210
  Points_Against = 175
  Games = 7

After this game:
  Points_For = 210 + 31 = 241
  Points_Against = 175 + 24 = 199
  Games = 8

Pyth_WinPct = PF^γ / (PF^γ + PA^γ)
Pyth_WinPct = 241^2.37 / (241^2.37 + 199^2.37)
Pyth_WinPct = 241^2.37 / (241^2.37 + 199^2.37)
Pyth_WinPct ≈ 0.625 (62.5%)

Pyth_ExpWins = 0.625 * 8 = 5.0
ActualWins = 6
Pyth_Luck = 6 - 5.0 = +1.0
```

**For BUF:**
```
After this game:
  Points_For = 195 + 24 = 219
  Points_Against = 185 + 31 = 216
  Games = 8

Pyth_WinPct = 219^2.37 / (219^2.37 + 216^2.37)
Pyth_WinPct ≈ 0.505 (50.5%)

Pyth_ExpWins = 0.505 * 8 = 4.04
ActualWins = 4
Pyth_Luck = 4 - 4.04 = -0.04
```

**Interpretation:**
- KAN's Pythagorean luck (+1.0) is higher than ELO luck (+0.527)
  - This suggests KAN is winning close games (good execution in key moments)
- BUF's Pythagorean luck (-0.04) is better than ELO luck (-0.527)
  - This suggests BUF's point differential is better than their record suggests

### 4.3 Volatility Updates

**For KAN:**
```
Elo_Change values for season (8 games):
  [2.4, -3.6, 5.0, -3.0, 6.0, -4.4, 5.6, 12.9]  # Last value is from this game

Volatility = std(Elo_Change) = std([2.4, -3.6, 5.0, -3.0, 6.0, -4.4, 5.6, 12.9])
Volatility ≈ 5.7

This game's change (12.9) increases volatility, but less dramatically than with higher K-factor.
```

**For BUF:**
```
Elo_Change values for season (8 games):
  [-3.0, 4.0, -2.4, 3.6, -5.0, 3.0, -4.0, -12.9]  # Last value is from this game

Volatility = std(Elo_Change) ≈ 5.6

This game's negative change increases volatility.
```

**Interpretation:**
- Both teams show high volatility (wild swings)
- This game contributes to that volatility pattern

### 4.4 Dynasty Score Updates

**For KAN:**
```
Elo_Post values for season (8 games):
  [1602.4, 1598.8, 1603.8, 1600.8, 1606.8, 1602.4, 1608.0, 1662.9]  # Last is from this game

DynastyScore = sum(max(0, Elo_Post - 1500)) / 100
DynastyScore = (102.4 + 98.8 + 103.8 + 100.8 + 106.8 + 102.4 + 108.0 + 162.9) / 100
DynastyScore = 885.9 / 100 = 8.86
```

**For BUF:**
```
Elo_Post values for season (8 games):
  [1577.0, 1581.0, 1578.6, 1582.2, 1577.2, 1580.2, 1576.2, 1567.1]  # Last is from this game

DynastyScore = (77.0 + 81.0 + 78.6 + 82.2 + 77.2 + 80.2 + 76.2 + 67.1) / 100
DynastyScore = 619.5 / 100 = 6.20
```

**Interpretation:**
- KAN's dynasty score (8.86) shows sustained dominance
- BUF's dynasty score (6.20) is lower, and this game drops it further

---

## Step 5: Update League-Wide Metrics

### 5.1 Calibration Bins

This game contributes to calibration analysis:

```
Predicted Win Probability: 0.673 (KAN)
Actual Outcome: 1.0 (KAN won)

This game goes into the bin: [0.65, 0.70] (approximately)

For this bin:
  Pwin_mean = average of all predicted probabilities in this bin
  Outcome_mean = average of all actual outcomes in this bin
  
If Outcome_mean ≈ Pwin_mean: Well calibrated
If Outcome_mean < Pwin_mean: Overconfident (predicted too high)
If Outcome_mean > Pwin_mean: Underconfident (predicted too low)
```

**Impact:**
- This game helps evaluate if the model is well-calibrated for 65-70% favorites
- If many 67% favorites win, it confirms calibration
- If they lose more often, it suggests overconfidence

### 5.2 Upset Analysis

**Upset Check:**
```
Underdog: BUF (lower pre-game ELO: 1580 vs 1650)
Underdog won: NO (BUF lost)
Upset: NO

Expected upset probability: min(A_Exp, B_Exp) = min(0.673, 0.327) = 0.327
Actual upset: 0 (didn't happen)

This contributes to:
  - UpsetRate calculation for ELO difference bin [100-125]
  - Upset Gap (actual - expected upsets)
```

**Impact:**
- This game is a "favorite won" scenario
- Contributes to upset statistics for games with ~125 point ELO difference
- Helps measure "Any Given Sunday" effect

### 5.3 Strength of Schedule Updates

**For KAN's Opponents:**
```
KAN's SoS metrics update:
  SoS_PreMean: Now includes BUF's pre-game ELO (1580)
  SoS_EndMean: Will update when season ends (uses BUF's final ELO)
  SoS_PreWeighted: Accounts for this being a home game (weight = 1.0)
  SoS_EndWeighted: Will update at season end
```

**For BUF's Opponents:**
```
BUF's SoS metrics update:
  SoS_PreMean: Now includes KAN's pre-game ELO (1650)
  SoS_EndMean: Will update when season ends
  SoS_PreWeighted: Accounts for this being an away game (weight = 1.15)
  SoS_EndWeighted: Will update at season end
```

**Impact:**
- KAN's schedule gets slightly easier (faced 1580 ELO team)
- BUF's schedule gets harder (faced 1650 ELO team, and it was away)

### 5.4 Parity Metrics (Season-Level)

**At Season End, This Game Affects:**
```
Dispersion: 
  - KAN's ELO increases (1662.9)
  - BUF's ELO decreases (1567.1)
  - This increases ELO spread, reducing parity

Rank Stability:
  - If KAN and BUF were ranked 1st and 2nd before, they stay that way
  - But the gap widens, affecting rank correlation

Upset Gap:
  - This was not an upset (favorite won)
  - Contributes to "fewer upsets than expected" if many favorites win

Forecast Entropy:
  - This game had entropy: -0.673*ln(0.673) - 0.327*ln(0.327) ≈ 0.64
  - Moderate entropy (not extreme probabilities)
  - Contributes to season entropy average
```

---

## Step 6: Monthly Aggregation Updates

### 6.1 Monthly ELO Data

**For October 2023:**
```
KAN's October games (including this one):
  - Last game ELO in October: 1662.9
  - Mean ELO in October: average of all October games
  - Rolling 3-month average updates
```

**Impact:**
- Monthly ELO data gets updated
- Rolling averages recalculate
- Tapestry plots will show this game's impact on KAN's trajectory

---

## Step 7: Conference and Division Updates

### 7.1 Conference Balance

**Both teams are AFC:**
```
AFC Mean ELO (after this game):
  - KAN: 1662.9 (increases AFC mean)
  - BUF: 1567.1 (decreases AFC mean)
  - Net effect on AFC mean depends on other teams
```

**Impact:**
- If KAN and BUF are top AFC teams, this game affects AFC dominance
- Conference balance metrics update

### 7.2 Division Rankings

**KAN is in AFC West, BUF is in AFC East:**
```
AFC West Mean ELO:
  - KAN's increase (1662.9) raises division mean
  - Affects division strength ranking

AFC East Mean ELO:
  - BUF's decrease (1567.1) lowers division mean
  - Affects division strength ranking
```

**Impact:**
- Division summaries update
- Division parity scores recalculate
- Division strength rankings may shift

---

## Step 8: Forecast Sharpness

### 8.1 Sharpness Metric

**This game contributes to:**
```
Season 2023 Sharpness:
  - Predicted probability: 0.673
  - This is a "strong call" (not near 50%)
  - Increases sharpness (std of probabilities)
  
If model makes many strong calls (high probabilities):
  - High sharpness = model is confident
  - Low sharpness = model is timid (clusters around 50%)
```

**Impact:**
- This game's 67.3% prediction contributes to season sharpness
- Helps evaluate if model is making strong calls or being too cautious

---

## Step 9: Season Reset (If Applicable)

**This game is mid-season, so no reset occurs.**

**If this were the first game of 2024 season:**
```
Season Reset (if enabled):
  λ = 0.15 (SEASON_RESET_LAMBDA)
  
  KAN's 2023 end ELO: 1700
  KAN's 2024 start ELO = (1 - 0.15) * 1700 + 0.15 * 1000
                        = 0.85 * 1700 + 150
                        = 1445 + 150
                        = 1595
  
  This prevents runaway drift and anchors to current era.
  With λ=0.15, teams retain 85% of their previous season's ELO.
```

---

## Summary: Complete Impact of This Single Game

### Direct Impacts:
1. **ELO Ratings**: KAN +12.9, BUF -12.9
2. **Win/Loss Records**: KAN 6-2, BUF 4-4
3. **Points**: KAN +31 PF, +24 PA; BUF +24 PF, +31 PA

### Metric Updates:
1. **Luck Index**: KAN +0.527, BUF -0.527
2. **Close-Game Luck**: Both teams' close game metrics update
3. **Pythagorean Luck**: KAN +1.0, BUF -0.04
4. **Volatility**: Both teams' volatility increases
5. **Dynasty Score**: KAN increases, BUF decreases
6. **SoS**: Both teams' strength of schedule updates
7. **Calibration**: Contributes to 65-70% favorite bin
8. **Upset Analysis**: Contributes to favorite-won statistics
9. **Parity Metrics**: Affects dispersion, entropy, upset gap
10. **Sharpness**: Contributes to forecast confidence metric

### Data Structure Updates:
1. **elo_games**: 1 new record
2. **timeline**: 2 new records (one per team)
3. **team_monthly_elo**: Monthly aggregations update
4. **season_team_summary**: Season totals update
5. **luck_index**: Luck metrics recalculate
6. **parity_metrics**: League parity recalculates
7. **division_summary**: Division metrics update
8. **conference_summary**: Conference balance updates

### Visualization Updates:
1. **League Tapestry**: KAN's line goes up, BUF's goes down
2. **Team Small Multiples**: Both teams' trajectories update
3. **Season Ladder**: Rankings may shift
4. **Luck Index Plot**: Bars update for both teams
5. **SoS vs ELO**: Points shift for both teams
6. **All other plots**: Indirectly affected by updated metrics

---

## Key Insights from This Example

1. **Close Games Matter**: Even expected wins can cause large ELO changes if the game is close (high MOV factor)

2. **Home Field Advantage**: The 55-point HFA made KAN's effective rating 1705 vs 1650, significantly affecting win probability

3. **Luck vs. Execution**: KAN's Pythagorean luck (+1.0) being higher than ELO luck (+0.527) suggests good execution in close games

4. **Volatility Impact**: ELO swings (like +12.9) increase team volatility, but with K=20 the changes are more moderate than with higher K-factors

5. **Cascading Effects**: One game affects multiple metrics, data structures, and visualizations throughout the system

6. **Context Matters**: The same 7-point win would have different impacts if:
   - It was a blowout (lower MOV factor)
   - The teams had different ELO ratings
   - It was an upset (underdog won)
   - It was early vs. late in the season

---

## Conclusion

A single NFL game triggers a cascade of calculations across the entire system. From immediate ELO updates to season-long metrics like parity and sharpness, every game contributes to the comprehensive analysis. This demonstrates the interconnected nature of the metrics and how they build upon each other to provide a complete picture of NFL team performance.

