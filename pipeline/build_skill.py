"""
Club skill, manager skill and organic growth for the Eredivisie and the
Eerste Divisie. Same definitions as the La Liga model, adapted to two leagues:

  * The value -> xPts trend line is fitted PER LEAGUE (the two leagues are
    different pools of teams and xPts is zero-sum inside each of them).
  * Everything is worked out per game and scaled to a full season, so the
    2026/27 season (only a few rounds played) is comparable with the finished ones.
  * The four reserve sides (Jong Ajax/PSV/AZ/Utrecht) are kept in the data but
    left out of the trend fit: their squad value is a talent pool, not a budget.

Outputs (data/processed/):
  team_seasons.csv      one row per league x season x club
  trend_params.csv      the fitted line per league
  manager_stints.csv    one row per manager x club x season (>= 5 games)
  organic_growth.csv    squad value created by players already in the pool, per club-season
  player_growth.csv     the same, one row per player
  manager_growth.csv    organic growth attributed to managers by share of games
  team_stats.csv        per-game team statistics (for / against), per league x season x club
"""
import os

import numpy as np
import pandas as pd

from names import RESERVE_SIDES, canon

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "..", "data", "raw")
OUT = os.path.join(BASE, "..", "data", "processed")

SEASON_LEN = {"Eredivisie": 34, "Eerste Divisie": 38}
MIN_STINT_GAMES = 5


def season_label(y):
    return f"{y % 100:02d}/{(y + 1) % 100:02d}"


def load_inputs():
    tm = pd.read_csv(os.path.join(OUT, "team_matches.csv"), parse_dates=["date"])
    clubs = pd.read_csv(os.path.join(RAW, "tm_clubs.csv"))
    clubs["club"] = clubs["tm_name"].map(canon)
    clubs["season_label"] = clubs["season"].map(season_label)
    mgr = pd.read_csv(os.path.join(RAW, "tm_managers.csv"))
    pv = pd.read_csv(os.path.join(RAW, "tm_player_values.csv"), dtype={"player_id": str})
    return tm, clubs, mgr, pv


# --------------------------------------------------------------------------- club seasons
def build_team_seasons(tm, clubs):
    g = (tm.groupby(["league", "season", "team"])
           .agg(games=("pts", "size"), pts=("pts", "sum"), gf=("gf", "sum"), ga=("ga", "sum"),
                xg=("xg_final", "sum"), xga=("xga_final", "sum"), xpts=("xpts", "sum"),
                est_share=("xg_source", lambda s: float((s == "estimated").mean())))
           .reset_index().rename(columns={"team": "club"}))
    g["season_len"] = g["league"].map(SEASON_LEN)
    g["partial"] = g["games"] < g["season_len"]
    g["is_reserve"] = g["club"].isin(RESERVE_SIDES)
    val = clubs[["league", "season_label", "club", "team_id", "squad_value_m"]].rename(columns={"season_label": "season"})
    g = g.merge(val, on=["league", "season", "club"], how="left")
    g["pts_pg"] = g["pts"] / g["games"]
    g["xpts_pg"] = g["xpts"] / g["games"]
    g["value_rank"] = g.groupby(["league", "season"])["squad_value_m"].rank(ascending=False, method="min")
    g["xg_source"] = np.where(g["est_share"] > 0.5, "estimated", "real")
    return g


def fit_trend(ts):
    rows = []
    for league, d in ts.groupby("league"):
        fit = d[(~d["partial"]) & (~d["is_reserve"]) & d["squad_value_m"].notna() & (d["squad_value_m"] > 0)]
        x, y = np.log(fit["squad_value_m"].values), fit["xpts_pg"].values
        m, b = np.polyfit(x, y, 1)
        resid = y - (m * x + b)
        rows.append({"league": league, "slope": m, "intercept": b, "n_team_seasons": len(fit),
                     "r2": 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                     "resid_sd_pg": resid.std(ddof=2), "season_len": SEASON_LEN[league]})
    return pd.DataFrame(rows)


def apply_trend(ts, trend):
    p = trend.set_index("league")
    ts = ts.copy()
    ts["predicted_xpts_pg"] = p.loc[ts["league"], "slope"].values * np.log(ts["squad_value_m"]) + p.loc[ts["league"], "intercept"].values
    ts["predicted_xpts"] = ts["predicted_xpts_pg"] * ts["season_len"]
    # club skill in xPts per FULL season (identical to xPts - predicted for a finished season)
    ts["club_skill"] = (ts["xpts_pg"] - ts["predicted_xpts_pg"]) * ts["season_len"]
    ts["pts_vs_budget"] = (ts["pts_pg"] - ts["predicted_xpts_pg"]) * ts["season_len"]
    ts["luck"] = (ts["pts_pg"] - ts["xpts_pg"]) * ts["season_len"]
    return ts


# --------------------------------------------------------------------------- managers
def build_stints(tm, ts, mgr, clubs):
    id_to_club = clubs.drop_duplicates("team_id").set_index("team_id")["club"]
    mg = mgr.copy()
    mg["club"] = mg["team_id"].map(id_to_club)
    mg["date_from"] = pd.to_datetime(mg["date_from"], format="%d/%m/%Y", errors="coerce")
    mg["date_to"] = pd.to_datetime(mg["date_to"], format="%d/%m/%Y", errors="coerce")
    mg = mg.dropna(subset=["date_from", "club"]).sort_values("date_from")

    m = tm.rename(columns={"team": "club"}).sort_values("date").copy()
    m = pd.merge_asof(m, mg[["club", "manager", "manager_id", "date_from", "date_to"]],
                      left_on="date", right_on="date_from", by="club", direction="backward")
    # a stint that already ended before the match means nobody (known) was in charge
    ended = m["date_to"].notna() & (m["date_to"] < m["date"])
    m.loc[ended, ["manager", "manager_id"]] = np.nan

    rows = []
    for (league, season, club), d in m.sort_values("date").groupby(["league", "season", "club"]):
        d = d.copy()
        d["block"] = ((d["manager_id"].fillna(-1) != d["manager_id"].fillna(-1).shift())).cumsum()
        for _, b in d.groupby("block"):
            if b["manager_id"].isna().all() or len(b) < MIN_STINT_GAMES:
                continue
            rows.append({"league": league, "season": season, "club": club,
                         "manager": b["manager"].iloc[0], "manager_id": int(b["manager_id"].iloc[0]),
                         "games": len(b), "start": b["date"].min(), "end": b["date"].max(),
                         "avg_xpts_pg": b["xpts"].mean(), "avg_pts_pg": b["pts"].mean(),
                         "avg_xg_pg": b["xg_final"].mean(), "avg_xga_pg": b["xga_final"].mean(),
                         "est_share": float((b["xg_source"] == "estimated").mean())})
    st = pd.DataFrame(rows)
    keep = ts[["league", "season", "club", "squad_value_m", "predicted_xpts_pg", "season_len", "is_reserve", "partial"]]
    st = st.merge(keep, on=["league", "season", "club"], how="left")
    st["stint_skill"] = (st["avg_xpts_pg"] - st["predicted_xpts_pg"]) * st["season_len"]
    st["xg_source"] = np.where(st["est_share"] > 0.5, "estimated", "real")
    return st


# --------------------------------------------------------------------------- organic growth
def build_organic_growth(pv, clubs):
    prev = (pv.groupby(["player_id", "season"])["value_m"].max().reset_index()
              .rename(columns={"value_m": "prev_value_m", "season": "prev_season"}))
    prev["season"] = prev["prev_season"] + 1
    cur = pv[["league", "season", "team_id", "player_id", "player_name", "position", "age", "value_m"]]
    pg = cur.merge(prev[["player_id", "season", "prev_value_m"]], on=["player_id", "season"], how="inner")
    pg["growth"] = pg["value_m"] - pg["prev_value_m"]
    id_to_club = clubs.drop_duplicates("team_id").set_index("team_id")["club"]
    pg["club"] = pg["team_id"].map(id_to_club)
    pg["season"] = pg["season"].map(season_label)
    pg = pg[pg["season"] >= "22/23"]
    # Transfermarkt has not revalued anyone yet for a season that has just started
    # (every value equals last season's), so "growth" would be a fake zero. Drop it.
    changed = pg.groupby("season")["growth"].apply(lambda s: (s != 0).mean())
    not_revalued = changed[changed < 0.05].index.tolist()
    if not_revalued:
        print(f"organic growth skipped for {not_revalued}: Transfermarkt has not revalued those seasons yet")
    pg = pg[~pg["season"].isin(not_revalued)]

    squad = cur.copy()
    squad["season"] = squad["season"].map(season_label)
    n_squad = squad.groupby(["league", "season", "team_id"]).size().rename("n_squad").reset_index()
    og = (pg.groupby(["league", "season", "club", "team_id"])
            .agg(growth_m=("growth", "sum"), n_matched=("growth", "size"),
                 prev_value_m=("prev_value_m", "sum"), value_m=("value_m", "sum"))
            .reset_index().merge(n_squad, on=["league", "season", "team_id"], how="left"))
    og["growth_pct"] = og["growth_m"] / og["prev_value_m"]
    return og, pg


def attribute_growth(stints, og):
    j = stints.groupby(["league", "season", "club", "manager", "manager_id"])["games"].sum().reset_index()
    tot = j.groupby(["league", "season", "club"])["games"].sum().rename("_tot").reset_index()
    j = j.merge(tot, on=["league", "season", "club"])
    j["share"] = j["games"] / j["_tot"]
    j = j.merge(og[["league", "season", "club", "growth_m"]], on=["league", "season", "club"], how="inner")
    j["attributed_m"] = j["growth_m"] * j["share"]
    return j.drop(columns="_tot")


# --------------------------------------------------------------------------- team stats
STAT_COLS = ["possession", "xg_final", "xg_np", "xgot", "shots", "shots_on_target", "shots_inside_box",
             "big_chances", "corners", "passes", "accurate_passes", "touches_opp_box", "tackles", "interceptions",
             "clearances", "fouls", "yellow_cards", "red_cards", "saves", "final_third_entries",
             "ball_recoveries", "dispossessed", "km_covered", "sprints", "opp_half_passes"]


def build_team_stats(tm):
    d = tm.copy()
    d["xg_final_against"] = d["xga_final"]
    cols = {}
    for c in STAT_COLS:
        if c in d.columns:
            cols[c] = "mean"
        if f"{c}_against" in d.columns:
            cols[f"{c}_against"] = "mean"
    agg = d.groupby(["league", "season", "team"]).agg(matches=("pts", "size"), **{k: (k, v) for k, v in cols.items()})
    sums = d.groupby(["league", "season", "team"])[["passes", "accurate_passes"]].sum(min_count=1)
    agg["pass_accuracy"] = sums["accurate_passes"] / sums["passes"]
    return agg.reset_index().rename(columns={"team": "club"})


def main():
    tm, clubs, mgr, pv = load_inputs()
    ts = build_team_seasons(tm, clubs)
    trend = fit_trend(ts)
    ts = apply_trend(ts, trend)
    stints = build_stints(tm, ts, mgr, clubs)
    og, pg = build_organic_growth(pv, clubs)
    mg = attribute_growth(stints, og)
    stats = build_team_stats(tm)

    ts = ts.merge(og[["league", "season", "club", "growth_m", "growth_pct", "n_matched", "n_squad"]],
                  on=["league", "season", "club"], how="left")
    ts.to_csv(os.path.join(OUT, "team_seasons.csv"), index=False)
    trend.to_csv(os.path.join(OUT, "trend_params.csv"), index=False)
    stints.to_csv(os.path.join(OUT, "manager_stints.csv"), index=False)
    og.to_csv(os.path.join(OUT, "organic_growth.csv"), index=False)
    pg.to_csv(os.path.join(OUT, "player_growth.csv"), index=False)
    mg.to_csv(os.path.join(OUT, "manager_growth.csv"), index=False)
    stats.to_csv(os.path.join(OUT, "team_stats.csv"), index=False)

    print(trend.round(3).to_string(index=False))
    print(f"team-seasons: {len(ts)}  stints: {len(stints)}  organic growth rows: {len(og)}  player rows: {len(pg)}")
    print("missing squad value:", int(ts["squad_value_m"].isna().sum()))


if __name__ == "__main__":
    main()
