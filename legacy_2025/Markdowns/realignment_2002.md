# 2002 NFL Realignment Table

## Overview
The 2002 NFL realignment restructured the league from **6 divisions** (3 per conference) to **8 divisions** (4 per conference), standardizing each division to **4 teams**. This realignment also coincided with the addition of the Houston Texans (HOU) as the 32nd team.

---

## Pre-2002 Structure (1995-2001)

### AFC Conference (3 divisions, 16 teams)

**AFC East (5 teams)**
- Buffalo Bills (BUF)
- Indianapolis Colts (IND)
- Miami Dolphins (MIA)
- New England Patriots (NWE)
- New York Jets (NYJ)

**AFC Central (6 teams)**
- Baltimore Ravens (BAL)
- Cincinnati Bengals (CIN)
- Cleveland Browns (CLE)
- Jacksonville Jaguars (JAX)
- Pittsburgh Steelers (PIT)
- Tennessee Titans (TEN)

**AFC West (5 teams)**
- Denver Broncos (DEN)
- Kansas City Chiefs (KAN)
- Las Vegas Raiders (LVR) - was Oakland
- Los Angeles Chargers (LAC) - was San Diego
- Seattle Seahawks (SEA)

### NFC Conference (3 divisions, 15 teams)

**NFC East (5 teams)**
- Arizona Cardinals (ARI)
- Dallas Cowboys (DAL)
- New York Giants (NYG)
- Philadelphia Eagles (PHI)
- Washington Commanders (WAS)

**NFC Central (5 teams)**
- Chicago Bears (CHI)
- Detroit Lions (DET)
- Green Bay Packers (GNB)
- Minnesota Vikings (MIN)
- Tampa Bay Buccaneers (TAM)

**NFC West (5 teams)**
- Atlanta Falcons (ATL)
- Carolina Panthers (CAR)
- New Orleans Saints (NOR)
- San Francisco 49ers (SFO)
- St. Louis Rams (STL) - now Los Angeles Rams (LAR)

---

## Post-2002 Structure (2002+)

### AFC Conference (4 divisions, 16 teams)

**AFC East (4 teams)**
- Buffalo Bills (BUF)
- Miami Dolphins (MIA)
- New England Patriots (NWE)
- New York Jets (NYJ)

**AFC North (4 teams)** *(formerly AFC Central)*
- Baltimore Ravens (BAL)
- Cincinnati Bengals (CIN)
- Cleveland Browns (CLE)
- Pittsburgh Steelers (PIT)

**AFC South (4 teams)** *(new division)*
- Houston Texans (HOU) - *expansion team*
- Indianapolis Colts (IND) - *moved from AFC East*
- Jacksonville Jaguars (JAX) - *moved from AFC Central*
- Tennessee Titans (TEN) - *moved from AFC Central*

**AFC West (4 teams)**
- Denver Broncos (DEN)
- Kansas City Chiefs (KAN)
- Las Vegas Raiders (LVR)
- Los Angeles Chargers (LAC)

### NFC Conference (4 divisions, 16 teams)

**NFC East (4 teams)**
- Dallas Cowboys (DAL)
- New York Giants (NYG)
- Philadelphia Eagles (PHI)
- Washington Commanders (WAS)

**NFC North (4 teams)** *(formerly NFC Central)*
- Chicago Bears (CHI)
- Detroit Lions (DET)
- Green Bay Packers (GNB)
- Minnesota Vikings (MIN)

**NFC South (4 teams)** *(new division)*
- Atlanta Falcons (ATL) - *moved from NFC West*
- Carolina Panthers (CAR) - *moved from NFC West*
- New Orleans Saints (NOR) - *moved from NFC West*
- Tampa Bay Buccaneers (TAM) - *moved from NFC Central*

**NFC West (4 teams)**
- Arizona Cardinals (ARI) - *moved from NFC East*
- St. Louis Rams (STL) - *moved to Los Angeles (LAR) in 2016*
- San Francisco 49ers (SFO)
- Seattle Seahawks (SEA) - *moved from AFC West*

---

## Key Changes Summary

### Major Team Movements

1. **Seattle Seahawks (SEA)**: AFC West → NFC West
   - Only team to switch conferences in the realignment

2. **Arizona Cardinals (ARI)**: NFC East → NFC West

3. **Indianapolis Colts (IND)**: AFC East → AFC South

4. **Tampa Bay Buccaneers (TAM)**: NFC Central → NFC South

5. **Atlanta, Carolina, New Orleans**: NFC West → NFC South

### New Additions

- **Houston Texans (HOU)**: Expansion team added to AFC South (2002)

### Division Renames

- **AFC Central** → **AFC North**
- **NFC Central** → **NFC North**

### Statistics

| Metric | Pre-2002 | Post-2002 |
|--------|----------|-----------|
| Total Teams | 31 | 32 |
| Divisions | 6 | 8 |
| Divisions per Conference | 3 | 4 |
| Teams per Division | 4-6 (variable) | 4 (standardized) |
| Conference Crossover | 0 | 1 (SEA) |

---

## Impact on ELO Analysis

The 2002 realignment is significant for ELO analysis because:

1. **Division-based metrics**: Division summaries and parity calculations need to account for the structural change
2. **Schedule changes**: Realignment affected intra-division and inter-division matchups
3. **Competitive balance**: The new structure may have affected competitive dynamics
4. **Season reset**: The realignment date (2002-09-01) can be used as a marker in visualizations

The code handles this by:
- Using season-based division mappings (post-2002 structure)
- Season reset mechanism prevents structural breaks from causing rating drift
- Division summaries are calculated per season, naturally handling the transition

