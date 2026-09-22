"""
Match Center data: builds tidy tables from the raw Sofascore match-detail dumps
(collect/js/sofascore_match_detail.js) saved under data/raw/match_detail/*.json
(each file: {"matches": {sofascore_event_id: {...}, ...}}).

Coordinates are in Sofascore's own system: 0-100 on both axes, attacking left-to-right
for the home team's own actions in `rating-breakdown` (pass/dribble/defensive/carry events)
and shots use their own 0-100 x / 0-100 y with a separate goal-mouth coordinate.

Outputs (data/processed/, parquet - the events/heatmap tables are 1M+ rows and compress far
better than CSV):
  match_center_matches.parquet       one row per match
  match_center_players.parquet       one row per player per match (aggregate stats)
  match_center_events.parquet        one row per pass/dribble/defensive-action/carry
  match_center_shots.parquet         one row per shot
  match_center_heatmap.parquet       one row per touch point
  match_center_momentum.parquet      one row per match-minute
  match_center_avgpos.parquet        one row per player's average position
"""
import glob
import json
import os

import pandas as pd

from names import canon

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "..", "data", "raw", "match_detail")
OUT = os.path.join(BASE, "..", "data", "processed")

DEF_TYPES = {"tackle", "interception", "clearance", "ball-recovery"}


def load_all():
    matches = {}
    for f in sorted(glob.glob(os.path.join(RAW, "*.json"))):
        payload = json.load(open(f, encoding="utf-8"))
        matches.update(payload["matches"])
    return matches


def safe_canon(name):
    try:
        return canon(name)
    except KeyError:
        return name


def main():
    raw = load_all()
    print(f"Loaded {len(raw)} matches from {RAW}")

    matches, players, events, shots, heatmap, momentum, avgpos = [], [], [], [], [], [], []

    for mid, m in raw.items():
        home, away = safe_canon(m["h"]), safe_canon(m["a"])
        date = pd.to_datetime(m["ts"], unit="s")
        matches.append({"match_id": mid, "league": m["league"], "season": m["season"], "round": m["round"],
                        "date": date, "home": home, "away": away, "home_score": m["hs"], "away_score": m["as"]})

        for minute, val in enumerate((m.get("momentum") or [])):
            momentum.append({"match_id": mid, "minute": val.get("minute", minute), "value": val.get("value")})

        for side, arr in ((m.get("avgpos") or {}).items() if m.get("avgpos") else []):
            if side not in ("home", "away"):
                continue
            for p in arr:
                avgpos.append({"match_id": mid, "side": side, "player_id": p["player"]["id"],
                               "player_name": p["player"]["name"], "shirt_number": p["player"].get("jerseyNumber"),
                               "avg_x": p.get("averageX"), "avg_y": p.get("averageY")})

        for s in (m.get("shots") or []):
            pc = s.get("playerCoordinates") or {}
            # Sofascore's shotmap stores x as distance FROM the goal being shot at (0 = on the goal line),
            # the opposite of rating-breakdown's pass/dribble/carry coordinates (0 = own goal, 100 = attacking
            # goal). Flip it here so every table in this pipeline shares one convention: x=100 is always the
            # goal the player is attacking.
            x = pc.get("x")
            shots.append({"match_id": mid, "player_id": s.get("player", {}).get("id"),
                          "player_name": s.get("player", {}).get("name"), "is_home": s.get("isHome"),
                          "minute": s.get("time"), "added_time": s.get("addedTime"), "shot_type": s.get("shotType"),
                          "situation": s.get("situation"), "body_part": s.get("bodyPart"),
                          "x": (100 - x) if x is not None else None, "y": pc.get("y"),
                          "xg": s.get("xg"), "xgot": s.get("xgot"), "is_own_goal": s.get("shotType") == "own-goal"})

        for p in m.get("players", []):
            st = dict(p.get("statistics") or {})
            rv = st.pop("ratingVersions", {}) or {}
            row = {"match_id": mid, "player_id": p["id"], "player_name": p["name"], "position": p.get("position"),
                  "is_home": p["isHome"], "substitute": p.get("substitute"),
                  "rating_original": rv.get("original"), "rating_alternative": rv.get("alternative")}
            row.update(st)
            players.append(row)

            for x, y in ((h["x"], h["y"]) for h in (p.get("heatmap") or [])):
                heatmap.append({"match_id": mid, "player_id": p["id"], "x": x, "y": y})

            ev = p.get("events") or {}
            for pas in ev.get("passes", []):
                sc, ec = pas.get("playerCoordinates", {}), pas.get("passEndCoordinates", {})
                # eventActionType is "pass", "cross" or "ball-touch" (a touch that isn't really a pass attempt,
                # e.g. a flick-on) - kept distinct so a pass map can show crosses differently.
                events.append({"match_id": mid, "player_id": p["id"], "event_type": pas.get("eventActionType", "pass"),
                              "x1": sc.get("x"), "y1": sc.get("y"), "x2": ec.get("x"), "y2": ec.get("y"),
                              "outcome": bool(pas.get("outcome")), "keypass": bool(pas.get("keypass")),
                              "long_ball": bool(pas.get("isLongBall"))})
            for d in ev.get("dribbles", []):
                sc = d.get("playerCoordinates", {})
                events.append({"match_id": mid, "player_id": p["id"], "event_type": "dribble", "x1": sc.get("x"), "y1": sc.get("y"),
                              "x2": None, "y2": None, "outcome": bool(d.get("outcome")), "keypass": False, "long_ball": False})
            for d in ev.get("defensive", []):
                sc = d.get("playerCoordinates", {})
                events.append({"match_id": mid, "player_id": p["id"], "event_type": d.get("eventActionType", "defensive"),
                              "x1": sc.get("x"), "y1": sc.get("y"), "x2": None, "y2": None, "outcome": bool(d.get("outcome")),
                              "keypass": False, "long_ball": False})
            for c in ev.get("ball-carries", []):
                sc, ec = c.get("playerCoordinates", {}), c.get("passEndCoordinates", {})
                events.append({"match_id": mid, "player_id": p["id"], "event_type": "carry", "x1": sc.get("x"), "y1": sc.get("y"),
                              "x2": ec.get("x"), "y2": ec.get("y"), "outcome": True, "keypass": False, "long_ball": False})

    os.makedirs(OUT, exist_ok=True)

    def save(rows, name, int_cols=(), float32_cols=()):
        df = pd.DataFrame(rows)
        for c in int_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int32")
        for c in float32_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
        df.to_parquet(os.path.join(OUT, f"{name}.parquet"), index=False)
        return df

    players_df = pd.DataFrame(players)
    avgpos_df = pd.DataFrame(avgpos)
    if not avgpos_df.empty:
        shirt = avgpos_df[["match_id", "player_id", "shirt_number"]].drop_duplicates(["match_id", "player_id"])
        players_df = players_df.merge(shirt, on=["match_id", "player_id"], how="left")

    save(matches, "match_center_matches", int_cols=["home_score", "away_score", "round"])
    save(players_df, "match_center_players")
    save(events, "match_center_events", float32_cols=["x1", "y1", "x2", "y2"])
    save(shots, "match_center_shots", float32_cols=["x", "y", "xg", "xgot"])
    save(heatmap, "match_center_heatmap", float32_cols=["x", "y"])
    save(momentum, "match_center_momentum", float32_cols=["value"])
    save(avgpos, "match_center_avgpos", float32_cols=["avg_x", "avg_y"])

    print(f"matches {len(matches)}  players {len(players)}  events {len(events)}  shots {len(shots)}  "
          f"heatmap {len(heatmap)}  momentum {len(momentum)}  avgpos {len(avgpos)}")


if __name__ == "__main__":
    main()
