import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import match_center as mc

st.set_page_config(page_title="Dutch Football Intelligence", layout="wide", page_icon="⚽")

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "data", "processed")

SEASONS = ["26/27", "25/26", "24/25", "23/24", "22/23"]
LEAGUES = ["Eredivisie", "Eerste Divisie"]
GREEN, RED, PURPLE, ORANGE, BLUE = "#2ecc71", "#e74c3c", "#8e44ad", "#e67e22", "#2980b9"


def season_fmt(s):
    return f"{s} (in progress)" if s == "26/27" else s


def money(v):
    return f"+€{v:.1f}m" if v >= 0 else f"-€{abs(v):.1f}m"


def short(name, limit=22):
    return name if len(name) <= limit else name[: limit - 1] + "…"


# ------------------------------------------------------------------ radar helpers
# (label, column, higher is better). "against" columns are flipped so that further out is always better,
# except in "Activity" where further out simply means more of it.
RADAR_SETS = {
    "Attack": [("xG", "xg_final", True), ("Shots", "shots", True), ("Shots on target", "shots_on_target", True),
               ("Shots inside box", "shots_inside_box", True), ("Big chances", "big_chances", True),
               ("Touches in opp. box", "touches_opp_box", True), ("Corners", "corners", True), ("xG on target", "xgot", True)],
    "Build-up & control": [("Possession", "possession", True), ("Passes", "passes", True), ("Pass accuracy", "pass_accuracy", True),
                           ("Opp.-half passes", "opp_half_passes", True), ("Final-third entries", "final_third_entries", True),
                           ("Ball recoveries", "ball_recoveries", True), ("Dispossessed", "dispossessed", False)],
    "Defence (chances conceded)": [("xG against", "xg_final_against", False), ("Shots against", "shots_against", False),
                                   ("SoT against", "shots_on_target_against", False),
                                   ("Box shots against", "shots_inside_box_against", False),
                                   ("Big chances against", "big_chances_against", False),
                                   ("Opp. touches in our box", "touches_opp_box_against", False),
                                   ("Corners against", "corners_against", False), ("xGOT against", "xgot_against", False)],
    "Activity": [("Tackles", "tackles", True), ("Interceptions", "interceptions", True), ("Clearances", "clearances", True),
                 ("Ball recoveries", "ball_recoveries", True), ("Saves", "saves", True), ("Fouls", "fouls", True),
                 ("Yellow cards", "yellow_cards", True), ("Km covered", "km_covered", True)],
}
RADAR_NOTES = {
    "Defence (chances conceded)": "Axes are flipped: further out = fewer chances conceded.",
    "Activity": "Volume, not quality: further out = more of it.",
    "Build-up & control": "Dispossessed is flipped: further out = loses the ball less.",
}
RADAR_COLORS = [BLUE, ORANGE, GREEN, PURPLE, RED, "#16a085"]


def _fmt(v):
    return f"{v:.2f}" if abs(v) < 5 else f"{v:.1f}"


def pct_score(pool, col, higher_better, by=None):
    """0-100 rank of each club on a statistic (100 = best / most): within the whole pool, or within each group
    of `by` (e.g. its own league and season) when given."""
    if by is None:
        s = pool[col]
        r = (s.rank(method="average") - 1) / max(s.notna().sum() - 1, 1) * 100
    else:
        g = pool.groupby(by)[col]
        r = (g.rank(method="average") - 1) / (g.transform("count") - 1).clip(lower=1) * 100
    return r if higher_better else 100 - r


def radar_axes(pool, spec, clubs):
    """Keep the axes every requested club has data for."""
    keep = []
    for label, col, hb in spec:
        if col not in pool.columns or pool[col].notna().sum() < 5:
            continue
        if all(pd.notna(pool.loc[pool.club == c, col]).any() for c in clubs):
            keep.append((label, col, hb))
    return keep


def draw_radar(ax, labels, series, title=None, fontsize=8):
    """series: [(name, values 0-100, color, filled)]. The dashed ring at 50 is the league median."""
    n = len(labels)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    closed = np.append(ang, ang[0])
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels([])
    ax.set_xticks(ang)
    ax.set_xticklabels(labels, fontsize=fontsize)
    ax.tick_params(axis="x", pad=9)
    ax.grid(color="#cccccc", lw=0.6)
    ax.spines["polar"].set_color("#cccccc")
    ax.plot(closed, [50] * (n + 1), color="#7f8c8d", lw=1.1, ls="--")
    for name, vals, color, filled in series:
        v = np.append(vals, vals[0])
        ax.plot(closed, v, color=color, lw=2, label=name)
        ax.scatter(ang, vals, color=color, s=18, zorder=4)
        if filled:
            ax.fill(closed, v, color=color, alpha=0.22)
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=22)


def radar_grid(pool_series, radar_sets=None, ncols=2):
    """Grid of radar sets (RADAR_SETS by default - 4 pillars, 2x2 - or any other {name: [(label, col,
    higher_better), ...]} dict; the grid is sized to fit however many are given). pool_series:
    [(name, pool_df, club, color)]. Values are 0-100 ranks in each pool."""
    radar_sets = radar_sets or RADAR_SETS
    nrows = -(-len(radar_sets) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.75 * nrows), subplot_kw={"projection": "polar"})
    axes_flat = np.array(axes).reshape(-1)
    for ax in axes_flat[len(radar_sets):]:
        ax.axis("off")
    for ax, (set_name, spec) in zip(axes_flat, radar_sets.items()):
        clubs_ok = radar_axes(pool_series[0][1], spec, [pool_series[0][2]])
        for _, pool, c, _col in pool_series[1:]:
            clubs_ok = [a for a in clubs_ok if a in radar_axes(pool, spec, [c])]
        if len(clubs_ok) < 3:
            ax.axis("off")
            ax.set_title(f"{set_name}: not enough data", fontsize=10)
            continue
        labels = []
        series = []
        for i, (name, pool, c, color) in enumerate(pool_series):
            vals = [pct_score(pool, col, hb)[pool.club == c].iloc[0] for _, col, hb in clubs_ok]
            series.append((name, np.array(vals), color, i == len(pool_series) - 1 or len(pool_series) == 1))
        for j, (label, col, _) in enumerate(clubs_ok):
            raws = [pool.loc[pool.club == c, col].iloc[0] for _, pool, c, _col in pool_series]
            labels.append(f"{label}\n" + " → ".join(_fmt(r) for r in raws))
        draw_radar(ax, labels, series, title=set_name)
    handles, names = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, names, loc="upper center", ncol=len(names), fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(left=0.1, right=0.9, top=0.92, bottom=0.05, wspace=0.65, hspace=0.42)
    return fig


# ------------------------------------------------------------------ data
@st.cache_data
def load():
    ts = pd.read_csv(os.path.join(PROC, "team_seasons.csv"))
    trend = pd.read_csv(os.path.join(PROC, "trend_params.csv"))
    stints = pd.read_csv(os.path.join(PROC, "manager_stints.csv"), parse_dates=["start", "end"])
    og = pd.read_csv(os.path.join(PROC, "organic_growth.csv"))
    pg = pd.read_csv(os.path.join(PROC, "player_growth.csv"))
    mg = pd.read_csv(os.path.join(PROC, "manager_growth.csv"))
    tstats = pd.read_csv(os.path.join(PROC, "team_stats.csv"))
    with open(os.path.join(PROC, "xg_model_report.json")) as f:
        report = json.load(f)
    return ts, trend, stints, og, pg, mg, tstats, report


ts, trend, stints, og, pg, mg, tstats, report = load()

# ------------------------------------------------------------------ sidebar
st.sidebar.title("Dutch Football Intelligence")
st.sidebar.caption(
    "Eredivisie and Eerste Divisie, 2022/23 – 2026/27: how clubs and managers perform against "
    "what their squads are worth, and how much value they create from the players they already have."
)
st.sidebar.divider()
league = st.sidebar.radio("League", LEAGUES)
season = st.sidebar.selectbox("Season", SEASONS, index=1, format_func=season_fmt)
clubs_now = sorted(ts[(ts.league == league) & (ts.season == season)]["club"].unique())
club = st.sidebar.selectbox("Club", clubs_now, key=f"club_{league}_{season}")
use_estimated = st.sidebar.checkbox(
    "Include seasons with estimated xG", value=True,
    help="Eerste Divisie 2022/23–2024/25 have no published xG anywhere. It is estimated from shot data "
         "(see the Framework tab for how accurate that is). Untick to pool only seasons with real xG.",
)
st.sidebar.divider()

with st.sidebar.expander("📖 New here? Start with the basics", expanded=False):
    st.markdown("""
**xG** (expected goals) rates every chance by how likely it was to score. Summed over a match it says
how good a team's chances really were, which is fairer than the scoreline.

**xPts** turns each match's xG into the points a team *deserved*. If actual points > xPts the team
was lucky; if lower, unlucky.

**Squad value** is the Transfermarkt market value of the whole squad: a stand-in for budget.

**Club skill** is how many xPts a club produced above (+) or below (−) what a squad of that value
typically produces in its league, scaled to a full season.

**Manager skill** is the same thing, but only counting the matches a manager was actually in charge of.

**Organic growth** is how much the market value of the club's players rose (or fell) over a season,
counting only players already in the two Dutch leagues the year before, so it is not bought value.
""")

# ------------------------------------------------------------------ helpers on the data
def allowed(df):
    """Optionally drop seasons whose xG is estimated (multi-season views)."""
    if use_estimated:
        return df
    return df[df["xg_source"] == "real"]


ts_all = ts.copy()
ts_league = ts[ts.league == league]
row = ts[(ts.league == league) & (ts.season == season) & (ts.club == club)]
row = row.iloc[0] if len(row) else None
table = ts_league[ts_league.season == season].copy()
table["gd"] = table["gf"] - table["ga"]
table = table.sort_values(["pts", "gd", "gf"], ascending=False).reset_index(drop=True)
table["position"] = table.index + 1

# ------------------------------------------------------------------ header
is_reserve = row is not None and bool(row["is_reserve"])
st.title(f"{club}{' (reserve side)' if is_reserve else ''}")
if row is not None:
    pos = int(table[table.club == club]["position"].iloc[0])
    st.caption(
        f"{league} · {season_fmt(season)} · {pos}{'st' if pos == 1 else 'nd' if pos == 2 else 'rd' if pos == 3 else 'th'} "
        f"of {len(table)} · {int(row['pts'])} pts after {int(row['games'])} games · xG source: **{row['xg_source']}**"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Squad value", f"€{row['squad_value_m']:.1f}m",
              help=f"Ranked {int(row['value_rank'])}/{len(table)} in the league this season.")
    c2.metric("xPts", f"{row['xpts']:.1f}",
              delta=f"{int(row['pts'])} actual pts ({row['pts'] - row['xpts']:+.1f} luck)", delta_color="off",
              help="Points the chances deserved. Actual points minus xPts is luck: positive = lucky.")
    c3.metric("Club skill", f"{row['club_skill']:+.1f} xPts",
              help="xPts above (+) or below (−) the value-predicted trend, per full season.")
    gm = row["growth_m"]
    c4.metric("Organic growth", money(gm) if pd.notna(gm) else "—",
              help="Market-value change of the players already in the Dutch pool the year before.")
    if row["partial"]:
        st.info(f"{season_fmt(season)} is still being played ({int(row['games'])} of {int(row['season_len'])} games). "
                "Skill values are per-game rates scaled to a full season, so they will move a lot early on.")
    if row["xg_source"] == "estimated":
        st.warning("This season's xG is **estimated** from shot data (no provider publishes xG for the Eerste Divisie "
                   "before 2025/26). Treat club and manager skill for it as approximate.")
st.divider()

tab_fw, tab_cl, tab_mg, tab_og, tab_td, tab_ss, tab_mc = st.tabs([
    "Framework", "Clubs", "Managers", "Organic Growth", "Team Data", "Season Stats", "Match Center",
])


# =================================================================== FRAMEWORK
with tab_fw:
    st.subheader("What the numbers measure")
    st.markdown("""
**Club skill** and **manager skill** answer one question: *given how much this squad is worth, did the team
play better or worse than teams with squads like it?* Results are measured in **xPts** (expected points from
chance quality), not real points, because real points contain luck.

1. Every match gets an xG for each side; a Poisson model turns the two xG values into expected points.
2. Per league, a trend line is fitted across every full team-season: `xPts per game = m × ln(squad value) + b`.
3. **Club skill** = (xPts per game − trend prediction) × season length.
4. **Manager skill** is the same, using only the games a manager was in charge of (a *stint*, minimum 5 games),
   so a manager is not credited or blamed for matches under a predecessor or successor.
5. **Organic growth** = sum over the club's players of (market value this season − market value last season),
   for players who were in either Dutch league the year before. It is split between managers by share of games.
""")
    st.subheader("Fitted trend lines")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, lg in zip(axes, LEAGUES):
        p_ = trend[trend.league == lg].iloc[0]
        d_ = ts[(ts.league == lg) & (~ts.partial) & (~ts.is_reserve)]
        ax.scatter(d_["squad_value_m"], d_["xpts_pg"] * p_["season_len"], color=BLUE if lg == "Eredivisie" else ORANGE,
                   s=30, alpha=0.65, edgecolors="white")
        xs_ = np.linspace(max(d_["squad_value_m"].min() * 0.9, 1), d_["squad_value_m"].max() * 1.1, 200)
        ax.plot(xs_, (p_["slope"] * np.log(xs_) + p_["intercept"]) * p_["season_len"], color="black", lw=1.6)
        ax.set_xscale("log")
        ax.set_title(f"{lg}: xPts/game = {p_['slope']:.2f} × ln(value €m) {p_['intercept']:+.2f}"
                     f"\nR² {p_['r2']:.2f} · {int(p_['n_team_seasons'])} team-seasons · residual SD {p_['resid_sd_pg']:.2f} xPts/game", fontsize=9)
        ax.set_xlabel("Squad value (€m, log scale)", fontsize=9)
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("xPts per full season", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Fitted on completed seasons only, without the four reserve sides (Jong Ajax, Jong PSV, Jong AZ, Jong Utrecht): "
               "their squad value is a pool of prospects rather than a budget, so it says little about expected results. "
               "They are still shown, marked as reserve sides.")

    st.subheader("Where the data comes from")
    st.markdown("""
- **Eredivisie matches, xG and team stats:** FotMob. Real xG in all 5 seasons (22/23 – 26/27).
- **Eerste Divisie matches, xG and team stats:** Sofascore. Real xG from 25/26; 22/23 – 24/25 estimated from shot data.
- **Squad and player market values:** Transfermarkt, per club and season, both leagues.
- **Managers:** Transfermarkt head-coach spells with appointment and leaving dates.
""")
    st.caption("Only regular-season matches count (no play-offs). FotMob and Sofascore publish identical numbers for the "
               "Eredivisie (checked on 96 team-matches: xG difference 0.000), so the choice of provider does not change the results.")

    st.subheader("How good is the estimated xG?")
    cs, ed = report["cross_season_summary"], report["eerste_divisie_training"]
    st.markdown(f"""
The estimator is a non-negative linear model on shots inside/outside the box, shots on target, blocked shots, big
chances created, corners and woodwork hits. It never sees goals, big chances scored/missed or the result, so
finishing luck cannot leak into the "expected" number.

**Test in the Eredivisie, where the true xG is known.** Train on one season, predict the others:
- Per team-match: R² **{cs['r2_match']:.2f}**, mean error **{cs['mae_match']:.2f} xG**.
- Per team-season xPts: correlation with the real thing **{cs['season_xpts_corr']:.3f}**, mean error
  **{cs['season_xpts_mae']:.1f} xPts** (real teams differ by about ±{cs['season_xpts_sd_real']:.0f} xPts).

**Test inside the Eerste Divisie itself** (2025/26 + 2026/27, out-of-fold): R² **{ed['cv_r2']:.2f}** per team-match;
season xPts correlation **{ed['season_xpts_corr']:.3f}**, mean error **{ed['season_xpts_mae']:.1f} xPts**
(spread ±{ed['season_xpts_sd_real']:.0f}).

So estimated seasons rank clubs reliably, but a single club's skill can be off by a couple of xPts.
""")


# =================================================================== CLUBS
with tab_cl:
    st.subheader("Clubs: who is beating the budget and who isn't")

    # ---- club ranking (pooled)
    st.markdown("#### Club ranking: average xPts above / below the value-predicted trend")
    pool_all = st.checkbox("All completed seasons (untick for the selected season only)", value=True, key="cl_pool")
    keep_res = st.checkbox("Include reserve sides", value=False, key="cl_res")
    base = allowed(ts_league[~ts_league.partial])
    if not pool_all:
        base = ts_league[ts_league.season == season]
    if not keep_res:
        base = base[~base.is_reserve]
    min_seasons = st.slider("Minimum seasons in the league", 1, 4, 2 if pool_all else 1, key="cl_min",
                            disabled=not pool_all)
    agg = (base.groupby("club").agg(seasons=("season", "nunique"), avg_skill=("club_skill", "mean"),
                                     above=("club_skill", lambda s: int((s > 0).sum())),
                                     best=("club_skill", "max"), worst=("club_skill", "min"),
                                     avg_value=("squad_value_m", "mean")).reset_index())
    agg = agg[agg.seasons >= (min_seasons if pool_all else 1)].sort_values("avg_skill", ascending=False)
    if agg.empty:
        st.info("No clubs meet that threshold.")
    else:
        fig, ax = plt.subplots(figsize=(9, max(4, len(agg) * 0.36)))
        a_ = agg[::-1].reset_index(drop=True)
        vals = a_["avg_skill"]
        bars = ax.barh(range(len(a_)), vals, color=[GREEN if v >= 0 else RED for v in vals], edgecolor="white", alpha=0.85)
        if pool_all:
            ax.hlines(range(len(a_)), a_["worst"], a_["best"], color="#34495e", lw=1.2, alpha=0.7, zorder=3)
            ax.scatter(a_["worst"], range(len(a_)), color="#34495e", s=14, zorder=4)
            ax.scatter(a_["best"], range(len(a_)), color="#34495e", s=14, zorder=4)
        ax.set_yticks(range(len(a_)))
        ax.set_yticklabels([f"{c} · €{v:.0f}m" + (f" · {n}s" if pool_all else "") for c, v, n in zip(a_["club"], a_["avg_value"], a_["seasons"])], fontsize=8)
        ax.axvline(0, color="grey", lw=0.9, ls="--")
        x_lab = a_["best"].max() + 0.5 if pool_all else None
        for b_, v, ab, se in zip(bars, vals, a_["above"], a_["seasons"]):
            if pool_all:
                ax.text(x_lab, b_.get_y() + b_.get_height() / 2, f"{v:+.1f}  ({ab}/{se} above trend)", va="center", ha="left", fontsize=8)
            else:
                ax.text(v + (0.25 if v >= 0 else -0.25), b_.get_y() + b_.get_height() / 2, f"{v:+.1f}",
                        va="center", ha="left" if v >= 0 else "right", fontsize=8)
        ax.margins(x=0.3 if pool_all else 0.12)
        ax.set_xlabel("Average club skill (xPts per season vs value-predicted trend)", fontsize=9)
        ax.set_title(f"{league}: club skill, " + ("completed seasons" if pool_all else season_fmt(season)), fontsize=11)
        ax.grid(axis="x", alpha=0.18)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Labels: club · average squad value" + (" · seasons in the league. Dark whiskers run from the club's worst to best season." if pool_all else "."))

    st.divider()

    # ---- justice table
    st.markdown(f"#### The Justice Table: {season_fmt(season)}")
    st.caption("Replace every team's actual points with the points its chances deserved (xPts). "
               "A club whose actual position is better than its deserved one was lucky.")
    j = table.copy()
    j["xpts_rank"] = j["xpts"].rank(ascending=False, method="min")
    j["luck_pts"] = j["pts"] - j["xpts"]
    j = j.sort_values("xpts_rank").reset_index(drop=True)
    cj1, cj2 = st.columns(2)
    with cj1:
        fig, ax = plt.subplots(figsize=(6, max(4, len(j) * 0.36)))
        y = np.arange(len(j))
        for i, r in j.iterrows():
            ax.plot([r["position"], r["xpts_rank"]], [i, i], color=RED if r["luck_pts"] > 0 else GREEN, lw=1.8, alpha=0.55)
        ax.scatter(j["position"], y, color=BLUE, s=60, zorder=3, edgecolors="white", label="Actual position")
        ax.scatter(j["xpts_rank"], y, color="#2c3e50", s=60, marker="D", zorder=3, edgecolors="white", label="Deserved (xPts)")
        ax.set_yticks(y)
        ax.set_yticklabels(j["club"], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("League position (1 = best)", fontsize=9)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(axis="x", alpha=0.15)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with cj2:
        jl = j.sort_values("luck_pts", ascending=False)
        fig, ax = plt.subplots(figsize=(6, max(4, len(j) * 0.36)))
        ax.barh(jl["club"][::-1], jl["luck_pts"][::-1], color=[RED if v > 0 else GREEN for v in jl["luck_pts"][::-1]], edgecolor="white")
        ax.axvline(0, color="grey", lw=0.8, ls="--")
        ax.set_xlabel("Luck = actual points − xPts", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # ---- value vs xpts scatter
    st.markdown("#### Squad value vs performance")
    full = ts_league[(~ts_league.partial) & (~ts_league.is_reserve)]
    p = trend[trend.league == league].iloc[0]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(full["squad_value_m"], full["xpts_pg"] * p["season_len"], color="#bdc3c7", s=36, alpha=0.7, label="Completed team-seasons")
    xs = np.linspace(max(full["squad_value_m"].min() * 0.9, 1), full["squad_value_m"].max() * 1.1, 200)
    ax.plot(xs, (p["slope"] * np.log(xs) + p["intercept"]) * p["season_len"], color="black", lw=1.6, label="Value-predicted trend")
    cur = ts_league[ts_league.season == season]
    cur = cur if keep_res else cur[~cur.is_reserve]
    ax.scatter(cur["squad_value_m"], cur["xpts_pg"] * p["season_len"], color=BLUE, s=60, edgecolors="white", label=season_fmt(season))
    for _, r in cur.iterrows():
        ax.annotate(short(r["club"], 14), (r["squad_value_m"], r["xpts_pg"] * p["season_len"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points", color="#1b4f72")
    if row is not None:
        ax.scatter([row["squad_value_m"]], [row["xpts_pg"] * p["season_len"]], color="#f1c40f", s=170, marker="*", edgecolors="black", zorder=5, label=club)
    ax.set_xscale("log")
    ax.set_xlabel("Squad value (€m, log scale)", fontsize=9)
    ax.set_ylabel("xPts per full season", fontsize=9)
    ax.set_title(f"{league}: squad value vs xPts (R² = {p['r2']:.2f})", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.18)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # ---- club history
    st.markdown(f"#### {club}: season by season")
    hist = ts[(ts.club == club)].copy()
    hist["order"] = hist["season"].map({s: i for i, s in enumerate(reversed(SEASONS))})
    hist = hist.sort_values("order")
    if hist.empty:
        st.info("No data for this club.")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        lab = [f"{r.season}\n{r.league[:3]}" + ("\n(so far)" if r.partial else "") for r in hist.itertuples()]
        vals = hist["club_skill"]
        axes[0].bar(range(len(hist)), vals, color=[GREEN if v >= 0 else RED for v in vals], edgecolor="white", width=0.6)
        for i, v in enumerate(vals):
            axes[0].text(i, v + (0.3 if v >= 0 else -0.3), f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8, fontweight="bold")
        axes[0].axhline(0, color="black", lw=1)
        axes[0].set_xticks(range(len(hist)))
        axes[0].set_xticklabels(lab, fontsize=8)
        axes[0].set_title("Club skill per season (xPts vs value trend)", fontsize=10)
        axes[1].plot(range(len(hist)), hist["squad_value_m"], marker="o", color=BLUE, lw=2)
        for i, v in enumerate(hist["squad_value_m"]):
            axes[1].annotate(f"€{v:.0f}m", (i, v), fontsize=8, xytext=(0, 6), textcoords="offset points", ha="center")
        axes[1].set_xticks(range(len(hist)))
        axes[1].set_xticklabels(lab, fontsize=8)
        axes[1].set_title("Squad value", fontsize=10)
        axes[1].grid(alpha=0.2)
        xi = np.arange(len(hist))
        wd = 0.27
        axes[2].bar(xi - wd, hist["pts_pg"], wd, color="#34495e", label="Actual pts / game")
        axes[2].bar(xi, hist["xpts_pg"], wd, color=BLUE, label="xPts / game (deserved)")
        axes[2].bar(xi + wd, hist["predicted_xpts_pg"], wd, color="#bdc3c7", label="Predicted from squad value")
        axes[2].set_xticks(xi)
        axes[2].set_xticklabels(lab, fontsize=8)
        axes[2].set_title("Points per game: actual vs deserved vs expected", fontsize=10)
        axes[2].legend(fontsize=7.5, loc="lower right")
        axes[2].grid(axis="y", alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# =================================================================== MANAGERS
with tab_mg:
    st.subheader("Managers: who is actually good at the job")
    st.markdown(
        "Same idea as club skill with one fix, **per-stint isolation**: only the matches a manager was in charge of count. "
        "A struggling team a manager inherited does not get pinned on them, and a hot start under a predecessor is not credited to them."
    )
    cm1, cm2, cm3 = st.columns(3)
    scope_season = cm1.radio("Seasons", ["Selected season", "All seasons"], index=1, horizontal=True, key="mg_seasons")
    scope_league = cm2.radio("Leagues", ["Selected league", "Both leagues"], index=1, horizontal=True, key="mg_leagues")
    keep_res_m = cm3.checkbox("Include reserve sides", value=False, key="mg_res")

    sm = stints.copy()
    sm = allowed(sm) if scope_season == "All seasons" else sm
    if scope_season == "Selected season":
        sm = sm[sm.season == season]
    if scope_league == "Selected league":
        sm = sm[sm.league == league]
    if not keep_res_m:
        sm = sm[~sm.is_reserve]
    max_g = int(max(sm["games"].sum() if len(sm) else 10, 10))
    default_g = 19 if scope_season == "All seasons" else 8
    min_games = st.slider("Minimum total games managed", 5, min(max_g, 152), min(default_g, min(max_g, 152)), key="mg_min")
    r2_e = trend.set_index("league")["r2"]
    st.caption("A partially played 2026/27 season is included when 'All seasons' is selected; those stints are short, so raise the "
               "minimum games to keep them out.")
    if scope_league == "Both leagues":
        st.caption(f"Each manager is measured against their own league's trend. Eerste Divisie values are more spread out because "
                   f"squad value explains less of the table there (R² {r2_e['Eerste Divisie']:.2f} vs {r2_e['Eredivisie']:.2f}), "
                   "so extremes at the top and bottom of a combined list are mostly Eerste Divisie names.")

    if sm.empty:
        st.info("No manager stints for this selection.")
    else:
        aggm = (sm.groupby(["manager_id", "manager"]).apply(lambda g: pd.Series({
            "stints": len(g), "games": int(g["games"].sum()),
            "clubs": ", ".join(sorted(set(g["club"]))),
            "avg_squad_value_m": np.average(g["squad_value_m"], weights=g["games"]),
            "skill": np.average(g["stint_skill"], weights=g["games"]),
        }), include_groups=False).reset_index())
        aggm = aggm[aggm.games >= min_games]
        if aggm.empty:
            st.info("No managers reach that number of games. Lower the slider.")
        else:
            n = min(12, len(aggm))
            best, worst = aggm.sort_values("skill", ascending=False).head(n), aggm.sort_values("skill").head(n)
            cb, cw = st.columns(2)
            for col, data, color, title in ((cb, best, GREEN, "Best (xPts above value trend)"), (cw, worst, RED, "Worst (xPts below value trend)")):
                with col:
                    st.markdown(f"**{title}**")
                    fig, ax = plt.subplots(figsize=(6, max(3.5, n * 0.42)))
                    y = np.arange(len(data))
                    ax.barh(y, data["skill"], color=color, edgecolor="white", alpha=0.88)
                    ax.set_yticks(y)
                    ax.set_yticklabels([f"{short(m, 20)} ({c.split(',')[0]})" for m, c in zip(data["manager"], data["clubs"])], fontsize=8)
                    ax.invert_yaxis()
                    ax.axvline(0, color="black", lw=1)
                    ax.set_xlabel("Manager skill (xPts per season vs value trend)", fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
            st.markdown("**Everyone in the selection**")
            fig, ax = plt.subplots(figsize=(10, 5.2))
            ax.scatter(aggm["games"], aggm["skill"], s=np.clip(aggm["avg_squad_value_m"], 5, 400) * 1.6 + 20,
                       c=[GREEN if v >= 0 else RED for v in aggm["skill"]], alpha=0.6, edgecolors="white")
            for r in pd.concat([best.head(6), worst.head(6)]).itertuples():
                ax.annotate(short(r.manager, 18), (r.games, r.skill), fontsize=7.5, xytext=(5, 3), textcoords="offset points")
            ax.axhline(0, color="black", lw=1)
            ax.set_xlabel("Games managed in the selection", fontsize=9)
            ax.set_ylabel("Manager skill (xPts per season vs value trend)", fontsize=9)
            ax.set_title("Skill vs sample size (bubble size = average squad value). More games = more trustworthy", fontsize=10)
            ax.grid(alpha=0.18)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.divider()
    st.markdown("#### Inspect one manager's stints")
    all_m = stints.drop_duplicates("manager_id").sort_values("manager")
    chosen = st.selectbox("Manager", all_m["manager"].tolist(), key="mg_pick")
    det = stints[stints.manager == chosen].copy()
    det["order"] = det["season"].map({s: i for i, s in enumerate(reversed(SEASONS))})
    det = det.sort_values(["order", "start"])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(max(12, len(det) * 2.2), 4.6))
    ax.bar(range(len(det)), det["stint_skill"], color=[GREEN if v >= 0 else RED for v in det["stint_skill"]], edgecolor="white", alpha=0.88, width=0.6)
    for i, v in enumerate(det["stint_skill"]):
        ax.text(i, v + (0.3 if v >= 0 else -0.3), f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(det)))
    ax.set_xticklabels([f"{r.club}\n{r.season}{' so far' if r.partial else ''} ({int(r.games)}g)" for r in det.itertuples()], fontsize=8)
    ax.axhline(0, color="black", lw=1.2)
    ax.set_ylabel("Stint skill (xPts per season vs value trend)", fontsize=9)
    ax.set_title(f"{chosen}: stint by stint", fontsize=11)
    ax.grid(axis="y", alpha=0.2)
    xi = np.arange(len(det))
    ax2.vlines(xi, det["predicted_xpts_pg"], det["avg_xpts_pg"], color="#95a5a6", lw=2)
    ax2.scatter(xi, det["predicted_xpts_pg"], color="#7f8c8d", s=70, marker="_", linewidths=3, label="Expected from squad value", zorder=3)
    ax2.scatter(xi, det["avg_xpts_pg"], color=BLUE, s=80, edgecolors="white", label="xPts / game under him", zorder=4)
    ax2.scatter(xi, det["avg_pts_pg"], color="#34495e", s=55, marker="D", edgecolors="white", label="Actual pts / game", zorder=4)
    ax2.set_xticks(xi)
    ax2.set_xticklabels([f"{r.club}\n{r.start:%b %y}–{r.end:%b %y}" for r in det.itertuples()], fontsize=8)
    ax.set_xlim(-0.7, max(len(det), 3) - 0.3)
    ax2.set_xlim(-0.7, max(len(det), 3) - 0.3)
    ax2.set_title("What the squad was worth vs what the team produced", fontsize=11)
    ax2.set_ylabel("Points per game", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# =================================================================== ORGANIC GROWTH
with tab_og:
    st.subheader("Organic growth: value created from players already in the league")
    st.markdown(
        "For every player in a club's squad who was already in the Eredivisie or Eerste Divisie the year before, take the change in "
        "Transfermarkt market value and add it up. It cannot be bought with transfer spending, so it is the best available proxy for "
        "player development. Players who arrived from abroad or from another league have no prior value in this pool and are left out."
    )
    ogl = og[og.league == league].merge(ts[["league", "season", "club", "is_reserve", "squad_value_m", "partial", "xg_source"]],
                                         on=["league", "season", "club"], how="left")

    # ---- club ranking
    st.markdown(f"#### Clubs: organic growth, {season_fmt(season)}")
    cur = ogl[ogl.season == season].copy()
    if cur.empty:
        st.info("No organic growth for this season yet: Transfermarkt has not revalued the squads for 2026/27, so every "
                "player's value still equals last season's. Pick 2025/26 or earlier." if season == "26/27"
                else "No growth data for this season.")
    else:
        cur = cur.sort_values("growth_m", ascending=False)
        fig, (ax, axp) = plt.subplots(1, 2, figsize=(13, max(4, len(cur) * 0.36)), sharey=True, gridspec_kw={"width_ratios": [1.5, 1]})
        c_ = cur[::-1].reset_index(drop=True)
        vals = c_["growth_m"]
        bars = ax.barh(range(len(c_)), vals, color=[PURPLE if v >= 0 else ORANGE for v in vals], edgecolor="white")
        ax.set_yticks(range(len(c_)))
        ax.set_yticklabels(c_["club"], fontsize=8)
        ax.axvline(0, color="grey", lw=0.9, ls="--")
        for b_, v, nm, ns in zip(bars, vals, c_["n_matched"], c_["n_squad"]):
            ax.text(v + (0.4 if v >= 0 else -0.4), b_.get_y() + b_.get_height() / 2, f"{money(v)}  ({int(nm)}/{int(ns)} players)",
                    va="center", ha="left" if v >= 0 else "right", fontsize=7.5)
        ax.margins(x=0.3)
        ax.set_xlabel("Organic squad value growth (€m)", fontsize=9)
        axp.barh(range(len(c_)), c_["growth_pct"] * 100, color=[PURPLE if v >= 0 else ORANGE for v in vals], edgecolor="white", alpha=0.6)
        axp.axvline(0, color="grey", lw=0.9, ls="--")
        axp.set_xlabel("Growth relative to last season's value of the same players (%)", fontsize=9)
        axp.tick_params(axis="y", left=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Brackets: players counted (already in the Dutch pool last year) out of squad size.")

    st.divider()
    st.markdown("#### Growth over the seasons: average per club")
    pooled = ogl[(~ogl.partial.fillna(False)) & (~ogl.is_reserve.fillna(False))]
    pooled = allowed(pooled)
    ag = pooled.groupby("club").agg(seasons=("season", "nunique"), avg_growth=("growth_m", "mean"), total=("growth_m", "sum")).reset_index()
    ag = ag[ag.seasons >= 2].sort_values("avg_growth", ascending=False)
    if not ag.empty:
        fig, ax = plt.subplots(figsize=(9, max(4, len(ag) * 0.34)))
        vals = ag["avg_growth"][::-1]
        ax.barh(ag["club"][::-1], vals, color=[PURPLE if v >= 0 else ORANGE for v in vals], edgecolor="white")
        ax.axvline(0, color="grey", lw=0.9, ls="--")
        ax.set_xlabel("Average organic growth per completed season (€m), clubs with 2+ seasons", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.markdown(f"#### {club}: who created the value ({season_fmt(season)})")
    pl = pg[(pg.league == league) & (pg.season == season) & (pg.club == club)].sort_values("growth", ascending=False)
    if pl.empty:
        st.info("No player growth data for this club and season.")
    else:
        top = pd.concat([pl.head(8), pl.tail(5)]).drop_duplicates("player_id")
        t_ = top[::-1].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(9, max(3, len(t_) * 0.42)))
        bars = ax.barh(range(len(t_)), t_["growth"], color=[PURPLE if v >= 0 else ORANGE for v in t_["growth"]], edgecolor="white")
        ax.set_yticks(range(len(t_)))
        ax.set_yticklabels([f"{n} ({p_}, {int(a) if pd.notna(a) else '?'})" for n, p_, a in zip(t_["player_name"], t_["position"], t_["age"])], fontsize=8)
        for b_, g, pv, nv in zip(bars, t_["growth"], t_["prev_value_m"], t_["value_m"]):
            ax.text(g + (0.1 if g >= 0 else -0.1), b_.get_y() + b_.get_height() / 2, f"€{pv:.1f}m → €{nv:.1f}m",
                    va="center", ha="left" if g >= 0 else "right", fontsize=7.5)
        ax.axvline(0, color="grey", lw=0.9)
        ax.margins(x=0.25)
        ax.set_xlabel("Change in market value (€m)", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.markdown("#### Managers: organic growth attributed by share of games")
    mgl = mg[mg.league == league].merge(ts[["league", "season", "club", "is_reserve", "xg_source", "partial"]], on=["league", "season", "club"], how="left")
    mgl = mgl[~mgl.is_reserve.fillna(False)]
    mgl = allowed(mgl[~mgl.partial.fillna(False)])
    agm = mgl.groupby(["manager_id", "manager"]).agg(seasons=("season", "nunique"), games=("games", "sum"), total=("attributed_m", "sum")).reset_index()
    agm = agm[agm.games >= 19]
    agm["per_season"] = agm["total"] / agm["seasons"]
    if agm.empty:
        st.info("Not enough completed seasons to attribute growth to managers yet.")
    else:
        n = min(10, len(agm))
        b, w = agm.sort_values("per_season", ascending=False).head(n), agm.sort_values("per_season").head(n)
        c1_, c2_ = st.columns(2)
        for col, data, color, title in ((c1_, b, PURPLE, "Most organic growth per season"), (c2_, w, ORANGE, "Least organic growth per season")):
            with col:
                st.markdown(f"**{title}**")
                fig, ax = plt.subplots(figsize=(6, max(3.5, n * 0.42)))
                ax.barh(range(len(data)), data["per_season"], color=color, edgecolor="white", alpha=0.88)
                ax.set_yticks(range(len(data)))
                ax.set_yticklabels([short(m, 22) for m in data["manager"]], fontsize=8)
                ax.invert_yaxis()
                ax.axvline(0, color="black", lw=1)
                ax.set_xlabel("Avg attributed growth per season (€m)", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        st.caption(f"{league}, completed seasons, managers with at least 19 games. Growth is a club-level number split by games managed, "
                   "so it says as much about the squad the manager inherited as about the manager.")


# =================================================================== TEAM DATA
STAT_LABELS = {
    "possession": "Possession (%)", "xg_final": "xG", "xg_np": "npxG (Eredivisie only)", "xgot": "xG on target",
    "shots": "Shots", "shots_on_target": "Shots on target", "shots_inside_box": "Shots inside box",
    "big_chances": "Big chances", "corners": "Corners", "passes": "Passes", "accurate_passes": "Accurate passes",
    "touches_opp_box": "Touches in opposition box", "tackles": "Tackles", "interceptions": "Interceptions",
    "clearances": "Clearances", "fouls": "Fouls", "yellow_cards": "Yellow cards", "red_cards": "Red cards", "saves": "Saves",
    "final_third_entries": "Final-third entries (Sofascore)", "ball_recoveries": "Ball recoveries (Sofascore)",
    "dispossessed": "Dispossessed (Sofascore)", "km_covered": "Km covered (Sofascore)", "sprints": "Sprints (Sofascore)",
    "opp_half_passes": "Passes in opposition half (FotMob)",
}

with tab_td:
    st.subheader(f"Team data: {league}, {season_fmt(season)}")
    st.caption("Per-game averages, built from every regular-season match. Eredivisie from FotMob, Eerste Divisie from Sofascore. "
               "Metrics only one provider publishes are left out of the radars for the other league.")
    td = tstats[(tstats.league == league) & (tstats.season == season)].copy()
    td = td.merge(ts[["league", "season", "club", "pts", "xpts", "club_skill", "squad_value_m", "is_reserve"]], on=["league", "season", "club"], how="left")
    avail = [c for c in STAT_LABELS if c in td.columns and td[c].notna().any()]
    order = SEASONS
    prev_season = order[order.index(season) + 1] if order.index(season) + 1 < len(order) else None

    # ---- club profile: four radars
    st.markdown(f"#### {club}: team profile")
    if club not in set(td["club"]):
        st.info("No team statistics for this club and season.")
    else:
        st.caption(f"Each axis is the club's rank among the {len(td)} {league} clubs this season: **100 = best / most, 0 = worst / least**. "
                   "The dashed ring is the league median. Under each axis name is the club's value per game.")
        fig = radar_grid([(club, td, club, BLUE)])
        st.pyplot(fig)
        plt.close(fig)
        st.caption(" ".join(f"**{k}:** {v}" for k, v in RADAR_NOTES.items()))

    # ---- compare clubs (any league, any season)
    st.divider()
    st.markdown("#### Compare clubs")
    st.caption("Pick any clubs from any season and either league, up to 6. The same club in different seasons works too.")
    all_ts = tstats.merge(ts[["league", "season", "club", "partial", "xg_source"]], on=["league", "season", "club"], how="left")
    all_ts["entry"] = all_ts["club"] + " · " + all_ts["season"] + " · " + all_ts["league"]
    all_ts["order"] = all_ts["season"].map({sn: i for i, sn in enumerate(SEASONS)})
    entries = all_ts.sort_values(["club", "order"])["entry"].tolist()
    me = f"{club} · {season} · {league}"
    leader = table[table.club != club]["club"].iloc[0] if len(table) > 1 else None
    leader_e = f"{leader} · {season} · {league}" if leader else None
    default = [e for e in [me, leader_e] if e in entries]
    cc1, cc2 = st.columns([2, 1])
    picked = cc1.multiselect("Clubs (club · season · league)", entries, default=default, max_selections=6,
                             key=f"td_cmp_{league}_{season}_{club}")
    set_pick = cc2.selectbox("Radar", list(RADAR_SETS), key="td_set")
    basis = st.radio("Rank each club", ["Within its own league and season", "Against all clubs in the selection's pool (both leagues, all seasons)"],
                     horizontal=True, key="td_basis",
                     help="First option: a club's rank among the clubs it actually played against that season, so a 90 means it was "
                          "near the top of its own league. Second option: every club-season in both leagues and all five seasons is "
                          "ranked together, which puts everything on one scale but mixes leagues of different strength and providers.")
    pool_all = all_ts.copy()
    pool_all["club"] = pool_all["entry"]          # radar helpers look clubs up by this column
    if len(picked) < 1:
        st.info("Pick at least one club.")
    else:
        spec = radar_axes(pool_all, RADAR_SETS[set_pick], picked)
        if len(spec) < 3:
            st.info("Not enough statistics shared by all the selected clubs (some are only published for one league or season).")
        else:
            by = ["league", "season"] if basis.startswith("Within") else None
            series = []
            for i, e in enumerate(picked):
                vals = np.array([pct_score(pool_all, col, hb, by=by)[pool_all.club == e].iloc[0] for _, col, hb in spec])
                series.append((e, vals, RADAR_COLORS[i % len(RADAR_COLORS)], len(picked) <= 3))
            fig, ax = plt.subplots(figsize=(8, 7.5), subplot_kw={"projection": "polar"})
            draw_radar(ax, [lab for lab, _, _ in spec], series,
                       title=f"{set_pick}: rank, 100 = best" + (" (within own league & season)" if by else " (all clubs pooled)"), fontsize=9)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8.5, frameon=False)
            fig.subplots_adjust(left=0.12, right=0.88, bottom=0.18)
            st.pyplot(fig)
            plt.close(fig)
            notes = [RADAR_NOTES[set_pick]] if set_pick in RADAR_NOTES else []
            flags = all_ts[all_ts["entry"].isin(picked)]
            if flags["partial"].fillna(False).any():
                notes.append("Includes a season still in progress: few games, so its values will move.")
            if (flags["xg_source"] == "estimated").any() and set_pick in ("Attack", "Defence (chances conceded)"):
                notes.append("Includes Eerste Divisie seasons with estimated xG.")
            if flags["league"].nunique() > 1:
                notes.append("Different leagues: statistics come from FotMob (Eredivisie) and Sofascore (Eerste Divisie), and the leagues differ in level.")
            if notes:
                st.caption(" ".join(notes))

    # ---- this season vs previous
    st.divider()
    st.markdown(f"#### {club}: {season_fmt(season)} against the previous season")
    prev_pool = tstats[(tstats.league == league) & (tstats.season == prev_season)] if prev_season else pd.DataFrame()
    if prev_season and club in set(prev_pool.get("club", [])) and club in set(td["club"]):
        st.caption(f"Ranks are within each season's own league table. Under each axis name: {prev_season} → {season} per-game value.")
        fig = radar_grid([(prev_season, prev_pool, club, "#95a5a6"), (season_fmt(season), td, club, BLUE)])
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.caption("No previous season in this league for this club.")

    # ---- xG for vs against
    if "xg_final" in td.columns:
        st.divider()
        st.markdown("#### Chance quality: xG created vs xG conceded (per game)")
        d = td.dropna(subset=["xg_final", "xg_final_against"]) if "xg_final_against" in td.columns else pd.DataFrame()
        if not d.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = [GREEN if v >= 0 else RED for v in d["club_skill"]]
            ax.scatter(d["xg_final"], d["xg_final_against"], c=colors, s=[170 if c == club else 80 for c in d["club"]], edgecolors="white")
            for _, r in d.iterrows():
                ax.annotate(short(r["club"], 14), (r["xg_final"], r["xg_final_against"]), fontsize=7.5, xytext=(4, 3), textcoords="offset points",
                            fontweight="bold" if r["club"] == club else "normal")
            ax.axvline(d["xg_final"].mean(), color="grey", lw=0.8, ls="--")
            ax.axhline(d["xg_final_against"].mean(), color="grey", lw=0.8, ls="--")
            ax.invert_yaxis()
            ax.set_xlabel("xG created per game", fontsize=9)
            ax.set_ylabel("xG conceded per game (axis inverted: up = better defence)", fontsize=9)
            ax.set_title("Green = above value trend, red = below (club skill)", fontsize=10)
            ax.grid(alpha=0.15)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ---- rank the league on one statistic
    st.divider()
    st.markdown("#### Rank the league on one statistic")
    if avail:
        cr1, cr2 = st.columns([2, 1])
        metric = cr1.selectbox("Statistic", avail, format_func=lambda c: STAT_LABELS[c], key="td_metric")
        side = cr2.radio("Side", ["For (the team)", "Against (the opponent)"], horizontal=True, key="td_side")
        col_ = metric if side.startswith("For") else ("xg_final_against" if metric == "xg_final" else f"{metric}_against")
        if col_ in td.columns and td[col_].notna().any():
            r = td[["club", col_]].dropna().sort_values(col_, ascending=False)
            fig, ax = plt.subplots(figsize=(9, max(4, len(r) * 0.34)))
            ax.barh(r["club"][::-1], r[col_][::-1], color=[BLUE if c != club else "#f1c40f" for c in r["club"][::-1]], edgecolor="white")
            ax.axvline(r[col_].mean(), color="grey", lw=0.9, ls="--")
            ax.set_xlabel(STAT_LABELS[metric] + (" per game" if side.startswith("For") else " conceded per game") + " (dashed = league average)", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.caption("That statistic has no 'against' version.")


# =================================================================== SEASON STATS + MATCH CENTER (shared load)
@st.cache_data
def load_match_center():
    return mc.load_match_center()


try:
    mc_matches, mc_players, mc_events, mc_shots, mc_heatmap, mc_momentum, mc_avgpos = load_match_center()
    MC_AVAILABLE = True
except FileNotFoundError:
    MC_AVAILABLE = False

SEASON_RADAR_SETS = {
    "Passing": [("Passes/game", "totalPass_pg", True), ("Pass accuracy", "pass_accuracy", True),
                ("Crosses/game", "totalCross_pg", True), ("Cross accuracy", "cross_accuracy", True),
                ("Key passes/game", "keyPass_pg", True), ("xA/game", "expectedAssists_pg", True)],
    "Duels & carrying": [("Duels won %", "duel_win_pct", True), ("Aerial duels won %", "aerial_win_pct", True),
                         ("Dribbles won %", "dribble_win_pct", True), ("Dribbles/game", "totalContest_pg", True),
                         ("Carries/game", "ballCarriesCount_pg", True), ("Progressive carries/game", "progressiveBallCarriesCount_pg", True)],
    "Defending": [("Tackles/game", "totalTackle_pg", True), ("Interceptions/game", "interceptionWon_pg", True),
                 ("Clearances/game", "totalClearance_pg", True), ("Recoveries/game", "ballRecovery_pg", True),
                 ("Fouls/game", "fouls_pg", False)],
    "Attacking output": [("Shots/game", "totalShots_pg", True), ("Big chances/game", "bigChanceCreated_pg", True),
                         ("Goals/game", "goals_pg", True), ("Touches/game", "touches_pg", True)],
    "Physical (tracking data)": [("Km covered/game", "kilometersCovered_pg", True), ("Sprints/game", "numberOfSprints_pg", True),
                                 ("High-speed running/game (km)", "metersCoveredHighSpeedRunningKm_pg", True),
                                 ("Sprint distance/game (km)", "metersCoveredSprintingKm_pg", True),
                                 ("Running distance/game (km)", "metersCoveredRunningKm_pg", True),
                                 ("Top speed this season (km/h)", "top_speed_max", True)],
}
SEASON_RANK_STATS = {lab: col for grp in SEASON_RADAR_SETS.values() for lab, col, _ in grp}

with tab_ss:
    st.subheader("Season Stats: whole-team season averages from the same Sofascore event data")
    st.caption("Every club's totals across its 2025/26 matches so far, turned into per-game rates and percentile "
               "ranks within its own league - passing, duels, dribbling/carrying, defending, attacking output and "
               "player-tracking (GPS) data. Team-level only (see Match Center below for single matches and players).")
    if not MC_AVAILABLE:
        st.warning("No match-detail data yet.")
    else:
        ssc1, ssc2 = st.columns(2)
        ss_league = ssc1.selectbox("League", sorted(mc_matches.league.unique()), key="ss_league")
        ss_season = ssc2.selectbox("Season", sorted(mc_matches[mc_matches.league == ss_league].season.unique(), reverse=True), key="ss_season")
        team_season = mc.team_season_totals(mc_players, mc_matches, ss_league, ss_season)
        team_season = team_season.rename(columns={"team": "club"})

        st.markdown("#### Club profile")
        ss_club = st.selectbox("Club", sorted(team_season.club), key="ss_club")
        st.caption(f"Each axis is this club's percentile rank among the {len(team_season)} {ss_league} clubs this season "
                   "(2025/26 so far): **100 = best / most, 0 = worst / least**. The dashed ring is the league median. "
                   "Under each axis name is the club's per-game value.")
        fig = radar_grid([(ss_club, team_season, ss_club, BLUE)], radar_sets=SEASON_RADAR_SETS, ncols=3)
        st.pyplot(fig)
        plt.close(fig)

        st.divider()
        st.markdown("#### Compare clubs")
        cmp_clubs = st.multiselect("Compare with", [c for c in team_season.club if c != ss_club], max_selections=3, key="ss_cmp")
        picked_clubs = [ss_club] + cmp_clubs
        set_pick = st.selectbox("Radar", list(SEASON_RADAR_SETS), key="ss_set")
        spec = radar_axes(team_season.assign(club=team_season.club), SEASON_RADAR_SETS[set_pick], picked_clubs)
        if len(spec) >= 3:
            series = []
            for i, cl in enumerate(picked_clubs):
                vals = np.array([pct_score(team_season, col, hb)[team_season.club == cl].iloc[0] for _, col, hb in spec])
                series.append((cl, vals, RADAR_COLORS[i % len(RADAR_COLORS)], len(picked_clubs) <= 3))
            fig, ax = plt.subplots(figsize=(7.5, 7), subplot_kw={"projection": "polar"})
            draw_radar(ax, [lab for lab, _, _ in spec], series, title=f"{set_pick}: league rank, 100 = best", fontsize=9)
            ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), fontsize=9, frameon=False)
            fig.subplots_adjust(left=0.18, right=0.82)
            st.pyplot(fig)
            plt.close(fig)

        st.divider()
        st.markdown(f"#### {ss_club}: season attack zones and shots")
        ids = mc.attach_team(mc_players, mc_matches)
        ids = ids[(ids.league == ss_league) & (ids.season == ss_season) & (ids.team == ss_club)]
        team_heat = mc_heatmap[mc_heatmap.player_id.isin(ids.player_id)]
        az1, az2 = st.columns(2)
        with az1:
            st.caption("Which side of the pitch this club's play happened on this season (own perspective - "
                       "checked against known left/right-footed fullbacks, so it's not a home/away artefact).")
            fig, ax = plt.subplots(figsize=(6.5, 1.4))
            mc.plot_width_thirds(ax, team_heat)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Same split, attacking third only (x ≥ 66.7) - where the ball actually ends up near goal.")
            fig, ax = plt.subplots(figsize=(6.5, 1.4))
            mc.plot_width_thirds(ax, team_heat, min_x=200 / 3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        with az2:
            club_matches = mc_matches[(mc_matches.league == ss_league) & (mc_matches.season == ss_season)
                                      & ((mc_matches.home == ss_club) | (mc_matches.away == ss_club))]
            club_shots = mc_shots[mc_shots.match_id.isin(club_matches.match_id)]
            club_shots = club_shots[club_shots.player_id.isin(ids.player_id.unique())]
            pitch, fig, ax = mc.new_pitch(figsize=(6.5, 5.5))
            mc.plot_shotmap(pitch, ax, club_shots, title=f"{ss_club}: every shot this season")
            st.pyplot(fig)
            plt.close(fig)

        st.divider()
        st.markdown("#### Rank the league on one stat")
        rank_lab = st.selectbox("Statistic", list(SEASON_RANK_STATS), key="ss_rank_stat")
        rank_col = SEASON_RANK_STATS[rank_lab]
        r = team_season[["club", rank_col]].dropna().sort_values(rank_col, ascending=False)
        fig, ax = plt.subplots(figsize=(9, max(4, len(r) * 0.34)))
        ax.barh(r["club"][::-1], r[rank_col][::-1], color=[BLUE if c != ss_club else "#f1c40f" for c in r["club"][::-1]], edgecolor="white")
        ax.axvline(r[rank_col].mean(), color="grey", lw=0.9, ls="--")
        ax.set_xlabel(rank_lab + " (dashed = league average)", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# =================================================================== MATCH CENTER
MAP_TYPES = {
    "Heatmap": ("heatmap", None),
    "Shots": ("shots", None),
    "Passes": ("events", mc.PASS_TYPES),
    "Dribbles": ("events", ["dribble"]),
    "Defensive actions": ("events", mc.DEF_TYPES),
    "Ball carries": ("events", ["carry"]),
}
TEAM_STAT_ROWS = [
    ("touches", "Touches", "{:.0f}"), ("totalPass", "Passes", "{:.0f}"),
    ("accuratePass", "Accurate passes", "{:.0f}"), ("totalCross", "Crosses", "{:.0f}"),
    ("keyPass", "Key passes", "{:.0f}"), ("expectedAssists", "xA", "{:.2f}"),
    ("duelWon", "Duels won", "{:.0f}"), ("aerialWon", "Aerial duels won", "{:.0f}"),
    ("wonContest", "Dribbles won", "{:.0f}"), ("totalTackle", "Tackles", "{:.0f}"),
    ("interceptionWon", "Interceptions", "{:.0f}"), ("totalClearance", "Clearances", "{:.0f}"),
    ("ballRecovery", "Ball recoveries", "{:.0f}"), ("fouls", "Fouls", "{:.0f}"),
]
KEY_STATS = [
    ("rating", "Rating", "{:.1f}"), ("totalPass", "Passes", "{:.0f}"), ("accuratePass", "Accurate passes", "{:.0f}"),
    ("keyPass", "Key passes", "{:.0f}"), ("expectedAssists", "xA", "{:.2f}"), ("totalTackle", "Tackles", "{:.0f}"),
    ("interceptionWon", "Interceptions", "{:.0f}"), ("duelWon", "Duels won", "{:.0f}"),
    ("aerialWon", "Aerial duels won", "{:.0f}"), ("kilometersCovered", "Km covered", "{:.1f}"),
]

with tab_mc:
    st.subheader("Match Center: any match, any player, evented data straight from Sofascore")
    if not MC_AVAILABLE:
        st.warning("No match-detail data yet. Run `pipeline/build_match_detail.py` after collecting matches "
                   "with `collect/js/sofascore_match_detail.js`.")
    else:
        st.caption(f"Currently covers **{mc_matches.season.iloc[0]}** for both leagues ({len(mc_matches)} matches). "
                   "Every pass, cross, dribble, defensive action and ball carry is the real tracked event (start/end "
                   "pitch coordinates and whether it succeeded), not an estimate - see the Framework tab for how this was collected.")

        c1, c2, c3 = st.columns([1, 1, 2])
        mc_league = c1.selectbox("League", sorted(mc_matches.league.unique()), key="mc_league")
        lg_matches = mc_matches[mc_matches.league == mc_league].sort_values("date")
        rounds = sorted(lg_matches["round"].unique())
        mc_round = c2.selectbox("Round", rounds, index=len(rounds) - 1, key="mc_round")
        round_matches = lg_matches[lg_matches["round"] == mc_round].copy()
        round_matches["label"] = (round_matches["home"] + " " + round_matches["home_score"].astype(str) + " - "
                                  + round_matches["away_score"].astype(str) + " " + round_matches["away"])
        mc_label = c3.selectbox("Match", round_matches["label"].tolist(), key="mc_match")
        mrow = round_matches[round_matches.label == mc_label].iloc[0]
        mid = mrow.match_id

        st.markdown(f"### {mrow.home} {int(mrow.home_score)} - {int(mrow.away_score)} {mrow.away}")
        st.caption(f"{mc_league} - round {int(mrow['round'])} - {mrow.date:%d %b %Y}")

        mplayers = mc_players[mc_players.match_id == mid].copy()
        mplayers["team"] = np.where(mplayers.is_home, mrow.home, mrow.away)
        mplayers["label"] = mplayers["player_name"] + " (" + mplayers["team"] + (
            mplayers["substitute"].map({True: ", sub", False: ""}).fillna("")) + ")"
        mheat = mc_heatmap[mc_heatmap.match_id == mid]
        mevents = mc_events[mc_events.match_id == mid]

        st.divider()
        st.markdown("#### Match overview")
        co1, co2 = st.columns(2)
        with co1:
            fig, ax = plt.subplots(figsize=(8, 3.2))
            mc.plot_momentum(ax, mc_momentum[mc_momentum.match_id == mid], mrow.home, mrow.away)
            ax.set_title("Match momentum (Sofascore's own xG-based model)", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        with co2:
            pitch, fig, ax = mc.new_pitch(figsize=(7.5, 5))
            mc.plot_avgpos_formation(pitch, ax, mc_avgpos[mc_avgpos.match_id == mid], mplayers, mrow.home, mrow.away)
            ax.set_title("Starting XI average positions (shirt numbers)", fontsize=10, pad=10)
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("##### Attack zones: which side of the pitch each team played through")
        st.caption("Own attacking perspective (not home/away), all touches and attacking-third-only, side by side per team.")
        az1, az2 = st.columns(2)
        for col, is_home, team_name in ((az1, True, mrow.home), (az2, False, mrow.away)):
            with col:
                st.caption(f"**{team_name}**")
                team_ids = mplayers.loc[mplayers.is_home == is_home, "player_id"]
                team_heat = mheat[mheat.player_id.isin(team_ids)]
                fig, ax = plt.subplots(figsize=(5.5, 1.3))
                mc.plot_width_thirds(ax, team_heat)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                fig, ax = plt.subplots(figsize=(5.5, 1.3))
                mc.plot_width_thirds(ax, team_heat, min_x=200 / 3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        st.markdown("##### Team stats: duels, passes and defending")
        st.caption("Summed from every player's individual match stats - the same numbers behind their rating.")
        home_tot, away_tot = mc.team_totals(mplayers)
        rows_present = [(k, lab, fmt) for k, lab, fmt in TEAM_STAT_ROWS if k in home_tot.index or k in away_tot.index]
        ncols = 3
        nrows = -(-len(rows_present) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 1.15 * nrows))
        for ax, (k, lab, fmt) in zip(np.array(axes).flat, rows_present):
            mc.plot_team_comparison(ax, home_tot.get(k), away_tot.get(k), mrow.home, mrow.away, lab, fmt=fmt)
        for ax in np.array(axes).flat[len(rows_present):]:
            ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.divider()
        st.markdown("#### Shot map")
        sh1, sh2 = st.columns(2)
        for col, is_home, team_name in ((sh1, True, mrow.home), (sh2, False, mrow.away)):
            with col:
                pitch, fig, ax = mc.new_pitch(figsize=(7, 6))
                mc.plot_shotmap(pitch, ax, mc_shots[(mc_shots.match_id == mid) & (mc_shots.is_home == is_home)],
                               title=f"{team_name}: shots (bubble size = xG)")
                st.pyplot(fig)
                plt.close(fig)

        st.divider()
        st.markdown("#### Player pitch maps")
        st.caption("Pick up to 4 players from this match (either team) to compare side by side.")
        default_players = mplayers.sort_values("rating", ascending=False)["label"].head(2).tolist()
        picked_labels = st.multiselect("Players", mplayers["label"].tolist(), default=default_players,
                                       max_selections=4, key=f"mc_players_{mid}")
        map_pick = st.selectbox("Map", list(MAP_TYPES), key="mc_maptype")
        source, kinds = MAP_TYPES[map_pick]

        if not picked_labels:
            st.info("Pick at least one player.")
        else:
            cols = st.columns(len(picked_labels))
            for col, lab in zip(cols, picked_labels):
                prow = mplayers[mplayers.label == lab].iloc[0]
                pid = prow.player_id
                shirt = f"#{int(prow.shirt_number)}" if pd.notna(prow.shirt_number) else ""
                with col:
                    st.caption(f"**{prow.player_name}** {shirt} - {prow.position} - {prow.team}")
                    pitch, fig, ax = mc.new_pitch(figsize=(5, 3.6))
                    if source == "heatmap":
                        mc.plot_heatmap(pitch, ax, mheat[mheat.player_id == pid])
                    elif source == "shots":
                        mc.plot_shotmap(pitch, ax, mc_shots[(mc_shots.match_id == mid) & (mc_shots.player_id == pid)])
                    else:
                        mc.plot_events(pitch, ax, mevents[mevents.player_id == pid], kinds, show_legend=False)
                    st.pyplot(fig)
                    plt.close(fig)
                    if map_pick == "Passes":
                        fig, ax = plt.subplots(figsize=(4.2, 1.5))
                        mc.plot_pass_thirds(ax, mevents[mevents.player_id == pid])
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

            st.markdown("##### Rating breakdown")
            rcols = st.columns(len(picked_labels))
            for col, lab in zip(rcols, picked_labels):
                prow = mplayers[mplayers.label == lab].iloc[0]
                with col:
                    st.caption(f"**{prow.player_name}** - rating {prow.rating:.1f}"
                              + (f" (alt. {prow.rating_alternative:.1f})" if pd.notna(prow.rating_alternative) else ""))
                    fig, ax = plt.subplots(figsize=(3.6, 2.2))
                    mc.plot_rating_breakdown(ax, prow)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

            st.markdown("##### Key numbers")
            picked_rows = mplayers[mplayers.label.isin(picked_labels)].set_index("label").loc[picked_labels].reset_index()
            colors = [BLUE, RED, GREEN, PURPLE][:len(picked_rows)]
            usable_stats = [(k, lab) for k, lab, fmt in KEY_STATS if k in picked_rows.columns and not picked_rows[k].isna().all()]
            ncols2 = 2
            nrows2 = -(-len(usable_stats) // ncols2)
            fig, axes = plt.subplots(nrows2, ncols2, figsize=(9, (0.5 + 0.4 * len(picked_rows)) * nrows2))
            for ax, (k, lab) in zip(np.array(axes).flat, usable_stats):
                rows = [(short(r.player_name, 16), getattr(r, k)) for r in picked_rows.itertuples()]
                mc.plot_stat_bars(ax, rows, lab, colors)
            for ax in np.array(axes).flat[len(usable_stats):]:
                ax.axis("off")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
