"""
Match Center: per-match, per-player pitch visuals built from Sofascore event data
(pipeline/build_match_detail.py). Coordinates are Sofascore's own 0-100 x 0-100 system,
already normalised so every player's own events run left (x=0, own goal) to right
(x=100, attacking goal) -- no flipping needed for a single player's own events.

Data covers the 2025/26 season of both leagues so far (686 matches). Older seasons
can be added the same way; see collect/js/sofascore_match_detail.js.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import Pitch

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "data", "processed")

GREEN, RED, BLUE, ORANGE, GOLD, GREY = "#2ecc71", "#e74c3c", "#2980b9", "#e67e22", "#f1c40f", "#7f8c8d"

EVENT_LABELS = {
    "pass": "Passes", "dribble": "Dribbles", "carry": "Ball carries", "tackle": "Tackles",
    "interception": "Interceptions", "clearance": "Clearances", "ball-recovery": "Ball recoveries",
    "block": "Blocks",
}
DEF_TYPES = ["tackle", "interception", "clearance", "ball-recovery", "block"]
SHOT_COLORS = {"goal": "#2ecc71", "save": "#2980b9", "block": "#e67e22", "post": "#f1c40f", "miss": "#95a5a6"}
SHOT_LABELS = {"goal": "Goal", "save": "Saved", "block": "Blocked", "post": "Hit the post", "miss": "Off target"}


def load_match_center():
    m = pd.read_parquet(os.path.join(PROC, "match_center_matches.parquet"))
    players = pd.read_parquet(os.path.join(PROC, "match_center_players.parquet"))
    events = pd.read_parquet(os.path.join(PROC, "match_center_events.parquet"))
    shots = pd.read_parquet(os.path.join(PROC, "match_center_shots.parquet"))
    heatmap = pd.read_parquet(os.path.join(PROC, "match_center_heatmap.parquet"))
    momentum = pd.read_parquet(os.path.join(PROC, "match_center_momentum.parquet"))
    avgpos = pd.read_parquet(os.path.join(PROC, "match_center_avgpos.parquet"))
    m["match_id"] = m["match_id"].astype(str)
    for d in (players, events, shots, heatmap, momentum, avgpos):
        d["match_id"] = d["match_id"].astype(str)
    return m, players, events, shots, heatmap, momentum, avgpos


def new_pitch(figsize=(7, 5), pitch_color="#f7f7f7", line_color="#b8b8b8", **kwargs):
    pitch = Pitch(pitch_type="opta", pitch_color=pitch_color, line_color=line_color, linewidth=1.1, **kwargs)
    fig, ax = pitch.draw(figsize=figsize)
    return pitch, fig, ax


def plot_events(pitch, ax, ev, kinds, title=None, show_legend=True):
    """ev: rows of match_center_events for one player. kinds: list of event_type to include."""
    d = ev[ev.event_type.isin(kinds)]
    if d.empty:
        ax.text(50, 50, "No data", ha="center", va="center", color=GREY, fontsize=11)
        if title:
            ax.set_title(title, fontsize=10.5, fontweight="bold")
        return
    has_end = d["x2"].notna()
    arrows, points = d[has_end], d[~has_end]
    if len(arrows):
        for ok, color, lab in ((True, GREEN, "Completed"), (False, RED, "Not completed")):
            sub = arrows[arrows.outcome == ok] if "carry" not in kinds or len(kinds) > 1 else arrows
            if "carry" in kinds and len(kinds) == 1:
                sub = arrows  # carries have no real fail state; always draw them the same way
                color = BLUE
            if sub.empty:
                continue
            pitch.arrows(sub.x1, sub.y1, sub.x2, sub.y2, ax=ax, color=color, width=1.6, headwidth=5, headlength=5,
                        alpha=0.75, label=lab if "carry" not in kinds else "Ball carries")
            if "carry" in kinds and len(kinds) == 1:
                break
    if len(points):
        for ok, color, lab in ((True, GREEN, "Won / completed"), (False, RED, "Lost / failed")):
            sub = points[points.outcome == ok]
            if sub.empty:
                continue
            pitch.scatter(sub.x1, sub.y1, ax=ax, color=color, s=70, alpha=0.85, edgecolors="white", linewidth=0.6, label=lab, zorder=3)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="upper center", bbox_to_anchor=(0.5, -0.02),
                      ncol=len(by_label), fontsize=8, frameon=False)
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold")


def plot_heatmap(pitch, ax, hm, title=None):
    if hm.empty:
        ax.text(50, 50, "No data", ha="center", va="center", color=GREY, fontsize=11)
    else:
        pitch.kdeplot(hm.x, hm.y, ax=ax, cmap="Greens", fill=True, levels=60, alpha=0.85, zorder=0)
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold")


def plot_shotmap(pitch, ax, sh, title=None, color_by_team=None):
    if sh.empty:
        ax.text(50, 50, "No shots", ha="center", va="center", color=GREY, fontsize=11)
        if title:
            ax.set_title(title, fontsize=10.5, fontweight="bold")
        return
    for kind, color in SHOT_COLORS.items():
        sub = sh[sh.shot_type == kind]
        if sub.empty:
            continue
        sizes = 80 + sub.xg.fillna(0.03) * 900
        pitch.scatter(sub.x, sub.y, ax=ax, s=sizes, color=color, alpha=0.8, edgecolors="white", linewidth=0.7,
                     label=SHOT_LABELS[kind], zorder=4 if kind == "goal" else 3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=len(SHOT_COLORS), fontsize=8, frameon=False)
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold")


def plot_momentum(ax, mom, home, away, home_color=BLUE, away_color=RED):
    if mom.empty:
        ax.text(0.5, 0.5, "No momentum data", ha="center", va="center", transform=ax.transAxes, color=GREY)
        return
    mom = mom.sort_values("minute")
    colors = [home_color if v >= 0 else away_color for v in mom.value]
    ax.bar(mom.minute, mom.value, color=colors, width=1.0)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Minute", fontsize=9)
    ax.set_yticks([])
    ymax = max(abs(mom.value.min()), abs(mom.value.max()), 10)
    ax.text(0, ymax * 0.92, f"◀ {away}", color=away_color, fontsize=9, fontweight="bold", ha="left")
    ax.text(0, -ymax * 0.92, f"{home} ▶", color=home_color, fontsize=9, fontweight="bold", ha="left")
    ax.set_ylim(-ymax, ymax)
    ax.grid(axis="x", alpha=0.15)


def plot_rating_breakdown(ax, row, color=BLUE):
    comps = [("Passing", "passValueNormalized"), ("Dribbling", "dribbleValueNormalized"),
             ("Defending", "defensiveValueNormalized"), ("Shooting", "shotValueNormalized"),
             ("Goalkeeping", "goalkeeperValueNormalized")]
    comps = [(lab, row.get(col)) for lab, col in comps if pd.notna(row.get(col))]
    if not comps:
        ax.axis("off")
        return
    labels = [c[0] for c in comps][::-1]
    vals = [c[1] for c in comps][::-1]
    colors = [GREEN if v >= 0 else RED for v in vals]
    ax.barh(labels, vals, color=colors, alpha=0.85, height=0.55)
    ax.axvline(0, color="black", lw=0.8)
    m = max(abs(min(vals)), abs(max(vals)), 0.1)
    ax.set_xlim(-m * 1.3, m * 1.3)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=9)
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)


def plot_avgpos_formation(pitch, ax, avgpos, players, home, away):
    """Starting-XI average positions on one pitch; the away team is flipped so it plays right-to-left."""
    starters = set(players.loc[~players.substitute.fillna(False), "player_id"])
    avgpos = avgpos[avgpos.player_id.isin(starters)]
    h = avgpos[avgpos.side == "home"]
    a = avgpos[avgpos.side == "away"]
    pos_lookup = players.set_index("player_id")["position"].to_dict()
    for d, flip, color, label in ((h, False, BLUE, home), (a, True, RED, away)):
        if d.empty:
            continue
        x = 100 - d.avg_x if flip else d.avg_x
        y = 100 - d.avg_y if flip else d.avg_y
        pitch.scatter(x, y, ax=ax, s=420, color=color, edgecolors="white", linewidth=1.3, zorder=3, alpha=0.9)
        for xi, yi, name in zip(x, y, d.player_name):
            ax.text(xi, yi, name.split()[-1][:3].upper(), ha="center", va="center", color="white", fontsize=7,
                    fontweight="bold", zorder=4)
    ax.text(2, 97, home, color=BLUE, fontsize=10, fontweight="bold")
    ax.text(98, 97, away, color=RED, fontsize=10, fontweight="bold", ha="right")
