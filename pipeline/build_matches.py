"""
Raw scrape -> one row per team per match (regular-season, finished matches only).

  Eredivisie      <- FotMob   (real xG in every season)
  Eerste Divisie  <- Sofascore (real xG from 25/26; earlier seasons get an
                     estimated xG later, see estimate_xg.py)

Output: data/processed/team_matches_raw.csv

Every statistic exists twice: "<stat>" (the team) and "<stat>_against" (the opponent).
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from names import canon

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "..", "data", "raw")
OUT = os.path.join(BASE, "..", "data", "processed")

FM_MAP = {
    "BallPossesion": "possession", "expected_goals": "xg", "expected_goals_non_penalty": "xg_np",
    "expected_goals_on_target": "xgot", "expected_goals_open_play": "xg_open", "expected_goals_set_play": "xg_set",
    "total_shots": "shots", "ShotsOnTarget": "shots_on_target", "ShotsOffTarget": "shots_off_target",
    "blocked_shots": "shots_blocked", "shots_inside_box": "shots_inside_box", "shots_outside_box": "shots_outside_box",
    "shots_woodwork": "woodwork", "big_chance": "big_chances", "big_chance_missed_title": "big_chances_missed",
    "corners": "corners", "Offsides": "offsides", "fouls": "fouls", "yellow_cards": "yellow_cards",
    "red_cards": "red_cards", "passes": "passes", "accurate_passes": "accurate_passes",
    "long_balls_accurate": "accurate_long_balls", "accurate_crosses": "accurate_crosses",
    "player_throws": "throw_ins", "touches_opp_box": "touches_opp_box", "own_half_passes": "own_half_passes",
    "opposition_half_passes": "opp_half_passes", "duel_won": "duels_won", "ground_duels_won": "ground_duels_won",
    "aerials_won": "aerials_won", "dribbles_succeeded": "dribbles_won", "matchstats.headers.tackles": "tackles",
    "interceptions": "interceptions", "clearances": "clearances", "keeper_saves": "saves",
}

SS_MAP = {
    "ballPossession": "possession", "expectedGoals": "xg", "expectedGoalsOnTarget": "xgot",
    "bigChanceCreated": "big_chances", "totalShotsOnGoal": "shots", "shotsOnGoal": "shots_on_target",
    "shotsOffGoal": "shots_off_target", "blockedScoringAttempt": "shots_blocked",
    "totalShotsInsideBox": "shots_inside_box", "totalShotsOutsideBox": "shots_outside_box",
    "bigChanceScored": "big_chances_scored", "bigChanceMissed": "big_chances_missed", "hitWoodwork": "woodwork",
    "cornerKicks": "corners", "fouls": "fouls", "offsides": "offsides", "yellowCards": "yellow_cards",
    "redCards": "red_cards", "passes": "passes", "accuratePasses": "accurate_passes",
    "accurateLongBalls": "accurate_long_balls", "accurateCross": "accurate_crosses", "throwIns": "throw_ins",
    "finalThirdEntries": "final_third_entries", "touchesInOppBox": "touches_opp_box",
    "dispossessed": "dispossessed", "duelWonPercent": "duels_won_pct", "aerialDuelsPercentage": "aerials_won_pct",
    "groundDuelsPercentage": "ground_duels_won_pct", "dribblesPercentage": "dribbles_won_pct",
    "wonTacklePercent": "tackles_won_pct", "totalTackle": "tackles", "interceptionWon": "interceptions",
    "ballRecovery": "ball_recoveries", "totalClearance": "clearances", "goalkeeperSaves": "saves",
    "goalsPrevented": "goals_prevented", "errorsLeadToShot": "errors_to_shot", "errorsLeadToGoal": "errors_to_goal",
    "highClaims": "high_claims", "kilometersCovered": "km_covered", "numberOfSprints": "sprints",
    "accurateThroughBall": "through_balls", "fouledFinalThird": "fouled_final_third",
}

# Event-type counts that Sofascore leaves out of the payload when the count is zero.
SS_ZERO_IF_MISSING = ["big_chances_scored", "big_chances_missed", "woodwork", "red_cards", "yellow_cards",
                      "errors_to_shot", "errors_to_goal", "offsides", "high_claims"]


def load_parts(prefix):
    rows, keys = [], None
    for f in sorted(glob.glob(os.path.join(RAW, f"{prefix}_part*.json"))):
        payload = json.load(open(f, encoding="utf-8"))
        rows += payload["rows"]
        keys = payload.get("keys", keys)
    return pd.DataFrame(rows), keys


def season_label(s):
    """'2022/2023' -> '22/23'; '22/23' stays."""
    return f"{s[2:4]}/{s[7:9]}" if len(s) == 9 else s


def stats_frame(stats, keys, mapping):
    """Wide frame: one row per match, columns home_<stat> / away_<stat>."""
    recs = []
    for r in stats.itertuples():
        d = {"_id": r[1]}
        for i, k in enumerate(keys):
            name = mapping.get(k)
            if name is None:
                continue
            d[f"home_{name}"], d[f"away_{name}"] = r.v[2 * i], r.v[2 * i + 1]
        recs.append(d)
    return pd.DataFrame(recs)


def to_team_rows(wide, id_col, stat_names):
    """Wide match frame -> long frame with one row per team."""
    common = ["league", "season", "round", "date", id_col]
    out = []
    for side, opp in (("home", "away"), ("away", "home")):
        d = wide[common].copy()
        d["team"] = wide[side].map(canon)
        d["opponent"] = wide[opp].map(canon)
        d["home"] = 1 if side == "home" else 0
        d["gf"] = wide["hg" if side == "home" else "ag"]
        d["ga"] = wide["ag" if side == "home" else "hg"]
        for s in stat_names:
            d[s] = wide[f"{side}_{s}"]
            d[f"{s}_against"] = wide[f"{opp}_{s}"]
        out.append(d)
    return pd.concat(out, ignore_index=True)


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---------------- Eredivisie (FotMob) ----------------
    fm_m, _ = load_parts("fm_matches")
    fm_s, fm_keys = load_parts("fm_stats")
    fm_m = fm_m[(fm_m.regular == 1) & (fm_m.finished == 1)].copy()
    fm_m["season"] = fm_m["season"].map(season_label)
    fm_m["date"] = pd.to_datetime(fm_m["utc"]).dt.tz_localize(None)
    fm_m = fm_m.rename(columns={"match_id": "src_id"})
    fm_w = stats_frame(fm_s, fm_keys, FM_MAP)
    fm_w["src_id"] = fm_w.pop("_id").astype(str)
    fm_m["src_id"] = fm_m["src_id"].astype(str)
    fm = fm_m.merge(fm_w, on="src_id", how="left")
    fm_stats = sorted({c[5:] for c in fm.columns if c.startswith("home_") and c not in ("home_id", "home")})
    ere = to_team_rows(fm, "src_id", fm_stats)
    ere["provider"] = "FotMob"

    # ---------------- Eerste Divisie (Sofascore) ----------------
    ss_m, _ = load_parts("ss_matches")
    ss_s, ss_keys = load_parts("ss_stats")
    ss_m = ss_m[ss_m.finished == 1].copy()
    ss_m["date"] = pd.to_datetime(ss_m["ts"], unit="s")
    # a pairing can only happen once in a regular season; drop anything repeated (play-offs etc.)
    ss_m = ss_m.sort_values(["season", "date"]).drop_duplicates(["season", "home_id", "away_id"], keep="first")
    ss_m = ss_m.rename(columns={"event_id": "src_id"})
    ss_w = stats_frame(ss_s[ss_s.has_stats == 1], ss_keys, SS_MAP)
    ss_w["src_id"] = ss_w.pop("_id")
    ss = ss_m.merge(ss_w, on="src_id", how="left")
    ss["src_id"] = ss["src_id"].astype(str)
    for name in SS_ZERO_IF_MISSING:
        for side in ("home", "away"):
            col = f"{side}_{name}"
            has_any = ss[f"{side}_shots"].notna()
            ss.loc[has_any, col] = ss.loc[has_any, col].fillna(0)
    ss_stats = sorted({c[5:] for c in ss.columns if c.startswith("home_") and c not in ("home_id", "home")})
    eer = to_team_rows(ss, "src_id", ss_stats)
    eer["provider"] = "Sofascore"

    tm = pd.concat([ere, eer], ignore_index=True)
    tm = tm.sort_values(["league", "season", "team", "date"]).reset_index(drop=True)
    tm["pts"] = np.where(tm.gf > tm.ga, 3, np.where(tm.gf == tm.ga, 1, 0))
    tm.to_csv(os.path.join(OUT, "team_matches_raw.csv"), index=False)

    print(tm.groupby(["league", "season"]).agg(rows=("team", "size"), teams=("team", "nunique"),
                                                 with_xg=("xg", lambda s: int(s.notna().sum()))))


if __name__ == "__main__":
    main()
