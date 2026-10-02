# Comprehensive Team Mapping Fix (1970+)

## Critical Fixes Applied

### 1. Baltimore (BAL) - FIXED ✅
**Issue:** BAL was incorrectly mapping pre-1996 to CLE (Browns), but should map pre-1984 to IND (Colts).

**History:**
- Baltimore Colts: pre-1984 → moved to Indianapolis
- Gap: 1984-1995 (no team in Baltimore)
- Baltimore Ravens: 1996+ (created from Browns relocation)

**Fix:**
- BAL pre-1984 → IND (Colts)
- BAL 1984-1995 → IND (error handling - no team existed)
- BAL 1996+ → BAL (Ravens)

### 2. St. Louis Cardinals (STL) - FIXED ✅
**Issue:** All STL codes were mapping to LAR (Rams), but Cardinals were also in St. Louis.

**History:**
- St. Louis Cardinals: 1960-1987 → moved to Phoenix/Arizona
- St. Louis Rams: 1995-2015 → moved to Los Angeles

**Fix:**
- STL pre-1995 → ARI (Cardinals)
- STL 1995+ → LAR (Rams)

### 3. Dallas (DAL) - VERIFIED ✅
**Note:** For 1970+ data, DAL is always Cowboys. No mapping needed.
- Dallas Cowboys: 1960+ (continuous)
- Dallas Texans (Chiefs): 1960-1962 → moved to KC in 1963
- Since data starts in 1970, DAL is always Cowboys

## Complete Mapping Summary (1970+)

### Static Mappings (No Date Logic)
- SDG → LAC (Chargers)
- OAK/RAI → LVR (Raiders)
- RAM → LAR (Rams)
- PHO → ARI (Cardinals)
- BOS → NWE (Patriots)
- OTI → TEN (Titans)

### Date-Based Mappings

#### St. Louis (STL)
- STL < 1995-01-01 → ARI (Cardinals)
- STL ≥ 1995-01-01 → LAR (Rams)

#### Houston (HOU)
- HOU < 1999-01-01 → TEN (Oilers)
- HOU ≥ 1999-01-01 → HOU (Texans)

#### Baltimore (BAL)
- BAL < 1984-01-01 → IND (Colts)
- BAL 1984-1995 → IND (error handling)
- BAL ≥ 1996-01-01 → BAL (Ravens)

#### Dallas (DAL)
- DAL → DAL (no change for 1970+ data)

## Teams Requiring No Mapping (1970+)

All other teams have continuous identities from 1970+:
- ATL, BUF, CAR, CHI, CIN, CLE, DEN, DET, GNB, IND, JAX, KAN, MIA, MIN, NOR, NYG, NYJ, PHI, PIT, SFO, SEA, TAM, WAS

## Files Updated

1. **NFL_ELO_organized.py** - Updated `canonical_franchise()` function
2. **NFL_Dash.html** - Updated `canonicalFranchise()` function
3. **team_relocations_table.tex** - Updated relocation table

## Verification Checklist

✅ Baltimore Colts (pre-1984) → Indianapolis Colts
✅ Baltimore Ravens (1996+) → Baltimore Ravens
✅ St. Louis Cardinals (pre-1995) → Arizona Cardinals
✅ St. Louis Rams (1995+) → Los Angeles Rams
✅ Houston Oilers (pre-1999) → Tennessee Titans
✅ Houston Texans (1999+) → Houston Texans
✅ All static mappings verified
✅ All date-based mappings verified

## Notes

- Data starts in 1970, so pre-1970 mappings are handled but may not be used
- The 1984-1995 gap for Baltimore is handled by mapping any BAL codes to IND (though this period should have no BAL codes)
- All mappings are now consistent between Python script and HTML dashboard

