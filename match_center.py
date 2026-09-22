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

GREEN, RED, BLUE, ORANGE, GOLD, GREY, PURPLE = "#2ecc71", "#e74c3c", "#2980b9", "#e67e22", "#f1c40f", "#7f8c8d", "#8e44ad"

# ---- passes/dribbles/carries/defensive events --------------------------------------------
EVENT_LABELS = {
    "pass": "Passes", "cross": "Crosses", "ball-touch": "Touches", "dribble": "Dribbles",
    "carry": "Ball carries", "tackle": "Tackles", "interception": "Interceptions",
    "clearance": "Clearances", "ball-recovery": "Ball recoveries", "block": "Blocks",
    "throw-in": "Throw-ins", "miss": "Miscontrols",
}
PASS_TYPES = ["pass", "cross", "ball-touch", "throw-in", "miss"]
DEF_TYPES = ["tackle", "interception", "clearance", "ball-recovery", "block"]
# marker per defensive event type (color is still won/lost)
DEF_MARKERS = {"tackle": "^", "interception": "s", "clearance": "D", "ball-recovery": "o", "block": "v"}
DRIBBLE_MARKER = "P"

# ---- shots ---------------------------------------------------------------------------------
SHOT_COLORS = {"goal": "#2ecc71", "save": "#2980b9", "block": "#e67e22", "post": "#f1c40f", "miss": "#95a5a6"}
SHOT_LABELS = {"goal": "Goal", "save": "Saved", "block": "Blocked", "post": "Hit the post", "miss": "Off target"}
SITUATION_MARKERS = {"regular": "o", "assisted": "o", "corner": "s", "free-kick": "^", "fast-break": "D", "penalty": "P"}
SITUATION_LABELS = {"regular": "Open play", "assisted": "Open play (assisted)", "corner": "Corner",
                    "free-kick": "Free kick", "fast-break": "Fast break", "penalty": "Penalty"}


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


def _legend(ax, handles_labels, ncol=None, y=-0.03, fontsize=8):
    handles, labels = handles_labels
    if not handles:
        return
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper center", bbox_to_anchor=(0.5, y),
              ncol=ncol or len(by_label), fontsize=fontsize, frameon=False)


def plot_events(pitch, ax, ev, kinds, title=None, show_legend=True):
    """ev: rows of match_center_events for one player. kinds: list of event_type to include.
    Passes/crosses/carries draw as arrows (green = completed, red = not, dashed = cross).
    Dribbles/defensive actions draw as points: colour = won/lost, marker shape = the specific type."""
    d = ev[ev.event_type.isin(kinds)]
    if d.empty:
        ax.text(50, 50, "No data", ha="center", va="center", color=GREY, fontsize=11)
        if title:
            ax.set_title(title, fontsize=10.5, fontweight="bold")
        return
    passlike = d[d.event_type.isin(["pass", "cross", "ball-touch", "throw-in", "miss"]) & d.x2.notna()]
    carries = d[(d.event_type == "carry") & d.x2.notna()]
    points = d[d.x2.isna()]

    for is_cross, style, lab_suffix in ((False, "-", ""), (True, "--", " (cross)")):
        sub_all = passlike[passlike.event_type.eq("cross") == is_cross]
        for ok, color in ((True, GREEN), (False, RED)):
            sub = sub_all[sub_all.outcome == ok]
            if sub.empty:
                continue
            pitch.arrows(sub.x1, sub.y1, sub.x2, sub.y2, ax=ax, color=color, width=1.6, headwidth=5, headlength=5,
                        alpha=0.8, linestyle=style, label=("Completed" if ok else "Not completed") + lab_suffix)
    if len(carries):
        pitch.arrows(carries.x1, carries.y1, carries.x2, carries.y2, ax=ax, color=BLUE, width=1.6, headwidth=5,
                    headlength=5, alpha=0.75, label="Ball carries")
    for etype in points.event_type.unique():
        marker = DEF_MARKERS.get(etype, DRIBBLE_MARKER if etype == "dribble" else "o")
        sub_type = points[points.event_type == etype]
        for ok, color in ((True, GREEN), (False, RED)):
            sub = sub_type[sub_type.outcome == ok]
            if sub.empty:
                continue
            pitch.scatter(sub.x1, sub.y1, ax=ax, color=color, s=75, alpha=0.88, marker=marker, edgecolors="white",
                         linewidth=0.6, label=f"{EVENT_LABELS.get(etype, etype)} ({'won' if ok else 'lost'})", zorder=3)
    if show_legend:
        _legend(ax, ax.get_legend_handles_labels(), fontsize=7.5)
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold")


def plot_pass_thirds(ax, ev, color=BLUE):
    """Horizontal-bar breakdown of where a player's passes/crosses started: defensive / middle / attacking third."""
    d = ev[ev.event_type.isin(["pass", "cross", "throw-in"])]
    if d.empty:
        ax.axis("off")
        return
    thirds = pd.cut(d.x1, [0, 100 / 3, 200 / 3, 100], labels=["Defensive third", "Middle third", "Attacking third"])
    pct = thirds.value_counts(normalize=True).reindex(["Defensive third", "Middle third", "Attacking third"]).fillna(0) * 100
    ax.barh([0], [100], color="#ecf0f1", height=0.6)
    left = 0
    shades = [0.45, 0.7, 1.0]
    for (lab, v), shade in zip(pct.items(), shades):
        ax.barh([0], [v], left=left, height=0.6, color=color, alpha=shade)
        if v > 6:
            ax.text(left + v / 2, 0, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        ax.text(left + v / 2, -0.62, lab.replace(" third", ""), ha="center", va="top", fontsize=7.5, color="#555")
        left += v
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.1, 0.6)
    ax.axis("off")


def plot_heatmap(pitch, ax, hm, title=None):
    if hm.empty:
        ax.text(50, 50, "No data", ha="center", va="center", color=GREY, fontsize=11)
    else:
        pitch.kdeplot(hm.x, hm.y, ax=ax, cmap="Greens", fill=True, levels=60, alpha=0.85, zorder=0)
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold")


def plot_team_zones(pitch, ax, hm_team, title=None, grid=(6, 4)):
    """Team-level 'where does play concentrate' grid, from the combined touch heatmap of every player."""
    if hm_team.empty:
        ax.text(50, 50, "No data", ha="center", va="center", color=GREY, fontsize=11)
        if title:
            ax.set_title(title, fontsize=10.5, fontweight="bold")
        return
    nx, ny = grid
    stats = pitch.bin_statistic(hm_team.x, hm_team.y, statistic="count", bins=(nx, ny))
    pitch.heatmap(stats, ax=ax, cmap="Greens", edgecolor="#f7f7f7", linewidth=1.5, alpha=0.9)
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold")


def plot_shotmap(pitch, ax, sh, title=None):
    if sh.empty:
        ax.text(50, 50, "No shots", ha="center", va="center", color=GREY, fontsize=11)
        if title:
            ax.set_title(title, fontsize=10.5, fontweight="bold")
        return
    for situ in sh.situation.dropna().unique():
        marker = SITUATION_MARKERS.get(situ, "o")
        sub_s = sh[sh.situation == situ]
        for kind, color in SHOT_COLORS.items():
            sub = sub_s[sub_s.shot_type == kind]
            if sub.empty:
                continue
            sizes = 70 + sub.xg.fillna(0.03) * 900
            pitch.scatter(sub.x, sub.y, ax=ax, s=sizes, color=color, marker=marker, alpha=0.82, edgecolors="white",
                         linewidth=0.7, label=SHOT_LABELS[kind], zorder=4 if kind == "goal" else 3)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    leg1 = ax.legend(by_label.values(), by_label.keys(), loc="upper center", bbox_to_anchor=(0.5, -0.03),
                     ncol=len(by_label), fontsize=7.5, frameon=False, title="Result", title_fontsize=7.5)
    ax.add_artist(leg1)
    shape_handles = [plt.Line2D([0], [0], marker=SITUATION_MARKERS.get(s, "o"), color="w", markerfacecolor=GREY,
                                markersize=8, label=SITUATION_LABELS.get(s, s)) for s in sh.situation.dropna().unique()]
    if shape_handles:
        ax.legend(handles=shape_handles, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=len(shape_handles),
                  fontsize=7.5, frameon=False, title="Situation (shape)", title_fontsize=7.5)
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
    """Starting-XI average positions on one pitch (shirt numbers), the away team flipped to play right-to-left."""
    starters = set(players.loc[~players.substitute.fillna(False), "player_id"])
    avgpos = avgpos[avgpos.player_id.isin(starters)]
    h = avgpos[avgpos.side == "home"]
    a = avgpos[avgpos.side == "away"]
    for d, flip, color, label in ((h, False, BLUE, home), (a, True, RED, away)):
        if d.empty:
            continue
        x = 100 - d.avg_x if flip else d.avg_x
        y = 100 - d.avg_y if flip else d.avg_y
        pitch.scatter(x, y, ax=ax, s=420, color=color, edgecolors="white", linewidth=1.3, zorder=3, alpha=0.9)
        for xi, yi, num in zip(x, y, d.shirt_number):
            lab = str(int(num)) if pd.notna(num) else "?"
            ax.text(xi, yi, lab, ha="center", va="center", color="white", fontsize=8.5, fontweight="bold", zorder=4)
    ax.text(2, 97, home, color=BLUE, fontsize=10, fontweight="bold")
    ax.text(98, 97, away, color=RED, fontsize=10, fontweight="bold", ha="right")


def team_totals(players_df):
    """Sum a match's per-player stats to two team rows (home/away). Used for the raw
    duels/passes/defending comparison in Match Overview - built from the exact same match_id,
    so it needs no join against a separately-collected team-stats table."""
    cols = ["totalPass", "accuratePass", "keyPass", "expectedAssists", "totalTackle", "interceptionWon",
            "totalClearance", "ballRecovery", "duelWon", "duelLost", "aerialWon", "aerialLost",
            "totalContest", "wonContest", "fouls", "touches", "kilometersCovered", "goals", "totalCross", "accurateCross"]
    cols = [c for c in cols if c in players_df.columns]
    g = players_df.groupby("is_home")[cols].sum(min_count=1)
    home = g.loc[True] if True in g.index else pd.Series(dtype=float)
    away = g.loc[False] if False in g.index else pd.Series(dtype=float)
    return home, away


def plot_team_comparison(ax, home_val, away_val, home_label, away_label, stat_label, home_color=BLUE, away_color=RED, fmt="{:.0f}"):
    """One stat, two bars (home vs away), for the Match Overview raw-data section."""
    vals = [away_val, home_val]
    names = [away_label, home_label]
    colors = [away_color, home_color]
    bars = ax.barh(names, [v if pd.notna(v) else 0 for v in vals], color=colors, alpha=0.88, height=0.55)
    mx = max([v for v in vals if pd.notna(v)] + [1])
    for b, v in zip(bars, vals):
        if pd.notna(v):
            ax.text(b.get_width() + mx * 0.03, b.get_y() + b.get_height() / 2, fmt.format(v), va="center", fontsize=8)
    ax.set_title(stat_label, fontsize=8.5, loc="left")
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=8)
    ax.margins(x=0.25)
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)


def plot_stat_bars(ax, rows, stat_label, colors):
    """rows: [(name, value), ...]. One horizontal bar per entry, for comparing players/teams on one stat."""
    names = [r[0] for r in rows][::-1]
    vals = [r[1] if pd.notna(r[1]) else 0 for r in rows][::-1]
    cs = colors[::-1]
    bars = ax.barh(names, vals, color=cs, alpha=0.88, height=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + max(vals + [1]) * 0.02, b.get_y() + b.get_height() / 2, f"{v:.2f}".rstrip("0").rstrip("."),
                va="center", fontsize=8)
    ax.set_xlabel(stat_label, fontsize=8.5)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.margins(x=0.18)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
