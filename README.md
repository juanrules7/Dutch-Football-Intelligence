# Dutch Football Intelligence

A Streamlit dashboard for the **Eredivisie** and the **Eerste Divisie**, 2022/23 – 2026/27.
It measures how clubs and managers perform against what their squads are worth, and how
much market value they create from players already in the two leagues.

Descriptive only: club skill, manager skill, organic growth, team data and a match-by-match
Match Center. There are no predictions and no sacking analysis.

## The metrics

| Metric | Definition |
|---|---|
| **xPts** | Points a team deserved: each match's xG for both sides goes through a Poisson model. |
| **Club skill** | (xPts per game − trend prediction) × season length. The trend is `xPts/game = m × ln(squad value) + b`, fitted per league on completed seasons. |
| **Manager skill** | The same, using only the games a manager was in charge of (a stint, minimum 5 games). |
| **Organic growth** | Sum over a club's players of (market value this season − last season) for players already in either Dutch league the year before. Split between managers by share of games. |

Reserve sides (Jong Ajax / PSV / AZ / Utrecht) are shown but left out of the trend fit.
2026/27 is still being played; skill is a per-game rate scaled to a full season and is flagged
"in progress". Organic growth for 2026/27 is not available because Transfermarkt has not
revalued the squads yet.

## Data sources

| Data | Source | Notes |
|---|---|---|
| Eredivisie matches, xG, team stats | FotMob | real xG every season |
| Eerste Divisie matches, xG, team stats | Sofascore | real xG from 2025/26 only |
| Eerste Divisie xG 2022/23 – 2024/25 | estimated | non-negative linear model on shot data, trained on 2025/26+; validated on the Eredivisie (season xPts correlation 0.989, mean error 1.5 xPts) |
| Squad and player market values, managers | Transfermarkt | |

FotMob and Sofascore publish identical numbers for the Eredivisie (xG difference 0.000 over
96 team-matches), so using different providers per league does not change the results.

## Match Center

A separate tab: pick any match from either league, pick up to 4 players from it, and compare
their heatmap, shot map, pass map (green/red arrows for completed/failed passes), dribbles,
defensive actions (tackles, interceptions, clearances, recoveries, blocks) and ball carries —
each the real tracked event with a pitch coordinate, not an estimate — plus Sofascore's own
player rating and its passing/dribbling/defending/shooting breakdown, match momentum, and the
starting-XI average positions. Coordinates are Sofascore's own 0–100 system, normalised in
`pipeline/build_match_detail.py` so x=100 is always the goal the player is attacking (shots
come from Sofascore inverted relative to every other event type, so they're flipped back).

Currently covers the **2025/26** season, both leagues (686 matches). Older seasons and the
Eredivisie's `AFC Ajax` → `Ajax` alias are handled the same way as everywhere else in this
pipeline (`pipeline/names.py`); extending coverage is just re-running the collector below with
a longer match list.

## Rebuilding

```
python collect/collect_transfermarkt.py      # squad values, player values, managers
# match data: run collect/js/fotmob_eredivisie.js in a fotmob.com tab and
# collect/js/sofascore_eerste_divisie.js in a sofascore.com tab, then move the results
# into data/raw/ with collect/ingest_tool_output.py
python pipeline/run_all.py                   # builds data/processed/*.csv

# Match Center: in a sofascore.com tab, set window.__matchList to [{league, season, round, id}, ...]
# (Sofascore event ids - see collect/js/sofascore_match_detail.js for how the existing 2025/26
# list was built) then run collect/js/sofascore_match_detail.js; pull window.__md.matches out in
# chunks (it's large - about 250-350 KB raw JSON per match) and save each chunk as its own file
# under data/raw/match_detail/ (any filename, must contain {"matches": {...}})
python pipeline/build_match_detail.py         # builds data/processed/match_center_*.parquet

streamlit run app.py
```

Sofascore returns 403 to plain Python requests, which is why all match data is fetched from
inside a browser tab. Everything under `data/raw/` (except `data/raw/match_detail/`, ~230 MB of
intermediate JSON - see `.gitignore`) and `data/processed/` is committed, so the app runs
without re-scraping.
