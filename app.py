import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

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

tab_fw, tab_cl, tab_mg, tab_og, tab_td = st.tabs([
    "Framework", "Clubs", "Managers", "Organic Growth", "Team Data",
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
    t = trend.copy()
    t["line"] = t.apply(lambda r: f"xPts/game = {r['slope']:.3f} × ln(value €m) + {r['intercept']:+.3f}", axis=1)
    t = t.rename(columns={"league": "League", "line": "Trend line", "n_team_seasons": "Team-seasons",
                          "r2": "R²", "resid_sd_pg": "Residual SD (xPts/game)"})
    st.dataframe(t[["League", "Trend line", "Team-seasons", "R²", "Residual SD (xPts/game)"]].round(3),
                 hide_index=True, width="stretch")
    st.caption("Fitted on completed seasons only, without the four reserve sides (Jong Ajax, Jong PSV, Jong AZ, Jong Utrecht): "
               "their squad value is a pool of prospects rather than a budget, so it says little about expected results. "
               "They are still shown, marked as reserve sides.")

    st.subheader("Where the data comes from")
    src = pd.DataFrame([
        ["Eredivisie", "FotMob", "Real xG, all 5 seasons (22/23 – 26/27)"],
        ["Eerste Divisie", "Sofascore", "Real xG from 25/26; 22/23–24/25 estimated from shot data"],
        ["Squad & player market values", "Transfermarkt", "Per club and season, both leagues"],
        ["Managers", "Transfermarkt", "Head-coach spells with appointment / leaving dates"],
    ], columns=["Data", "Source", "Coverage"])
    st.dataframe(src, hide_index=True, width="stretch")
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
        vals = agg["avg_skill"][::-1]
        bars = ax.barh(agg["club"][::-1], vals, color=[GREEN if v >= 0 else RED for v in vals], edgecolor="white")
        ax.axvline(0, color="grey", lw=0.9, ls="--")
        for b_, v in zip(bars, vals):
            ax.text(v + (0.25 if v >= 0 else -0.25), b_.get_y() + b_.get_height() / 2, f"{v:+.1f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=8)
        ax.margins(x=0.12)
        ax.set_xlabel("Average club skill (xPts per season vs value-predicted trend)", fontsize=9)
        ax.set_title(f"{league}: club skill, " + ("completed seasons" if pool_all else season_fmt(season)), fontsize=11)
        ax.grid(axis="x", alpha=0.18)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        show = agg.rename(columns={"club": "Club", "seasons": "Seasons", "avg_skill": "Avg skill", "above": "Seasons above trend",
                                   "best": "Best season", "worst": "Worst season", "avg_value": "Avg squad value (€m)"})
        st.dataframe(show.round(1), hide_index=True, width="stretch")

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
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
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
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        h = hist[["season", "league", "games", "pts", "xpts", "luck", "squad_value_m", "predicted_xpts", "club_skill", "xg_source"]].copy()
        h.columns = ["Season", "League", "Games", "Pts", "xPts", "Luck (pts − xPts)", "Squad value (€m)", "Predicted xPts", "Club skill", "xG source"]
        st.dataframe(h.round(1), hide_index=True, width="stretch")


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
            full_tab = aggm.sort_values("skill", ascending=False).rename(columns={
                "manager": "Manager", "stints": "Stints", "games": "Games", "clubs": "Clubs",
                "avg_squad_value_m": "Avg squad value (€m)", "skill": "Skill (xPts/season)"})
            st.dataframe(full_tab.drop(columns="manager_id").round(1), hide_index=True, width="stretch")

    st.divider()
    st.markdown("#### Inspect one manager's stints")
    all_m = stints.drop_duplicates("manager_id").sort_values("manager")
    chosen = st.selectbox("Manager", all_m["manager"].tolist(), key="mg_pick")
    det = stints[stints.manager == chosen].copy()
    det["order"] = det["season"].map({s: i for i, s in enumerate(reversed(SEASONS))})
    det = det.sort_values(["order", "start"])
    fig, ax = plt.subplots(figsize=(max(7, len(det) * 1.3), 4.6))
    ax.bar(range(len(det)), det["stint_skill"], color=[GREEN if v >= 0 else RED for v in det["stint_skill"]], edgecolor="white", alpha=0.88, width=0.6)
    for i, v in enumerate(det["stint_skill"]):
        ax.text(i, v + (0.3 if v >= 0 else -0.3), f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(det)))
    ax.set_xticklabels([f"{r.club}\n{r.season}{' so far' if r.partial else ''} ({int(r.games)}g)" for r in det.itertuples()], fontsize=8)
    ax.axhline(0, color="black", lw=1.2)
    ax.set_ylabel("Stint skill (xPts per season vs value trend)", fontsize=9)
    ax.set_title(f"{chosen}: stint by stint", fontsize=11)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    dd = det[["season", "league", "club", "games", "start", "end", "avg_xpts_pg", "avg_pts_pg", "squad_value_m", "stint_skill", "xg_source"]].copy()
    dd["start"], dd["end"] = dd["start"].dt.date, dd["end"].dt.date
    dd.columns = ["Season", "League", "Club", "Games", "From", "To", "xPts/game", "Pts/game", "Squad value (€m)", "Skill", "xG source"]
    st.dataframe(dd.round(2), hide_index=True, width="stretch")


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
        fig, ax = plt.subplots(figsize=(9, max(4, len(cur) * 0.36)))
        vals = cur["growth_m"][::-1]
        bars = ax.barh(cur["club"][::-1], vals, color=[PURPLE if v >= 0 else ORANGE for v in vals], edgecolor="white")
        ax.axvline(0, color="grey", lw=0.9, ls="--")
        for b_, v in zip(bars, vals):
            ax.text(v + (0.4 if v >= 0 else -0.4), b_.get_y() + b_.get_height() / 2, money(v), va="center",
                    ha="left" if v >= 0 else "right", fontsize=8)
        ax.margins(x=0.16)
        ax.set_xlabel("Organic squad value growth (€m)", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        t = cur[["club", "growth_m", "growth_pct", "n_matched", "n_squad", "prev_value_m", "value_m"]].copy()
        t["growth_pct"] = t["growth_pct"] * 100
        t.columns = ["Club", "Growth (€m)", "Growth (%)", "Players counted", "Squad size", "Their value last season (€m)", "Their value now (€m)"]
        st.dataframe(t.round(1), hide_index=True, width="stretch")

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
        fig, ax = plt.subplots(figsize=(8, max(3, len(top) * 0.4)))
        ax.barh(top["player_name"][::-1], top["growth"][::-1], color=[PURPLE if v >= 0 else ORANGE for v in top["growth"][::-1]], edgecolor="white")
        ax.axvline(0, color="grey", lw=0.9)
        ax.set_xlabel("Change in market value (€m)", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        tp = pl[["player_name", "position", "age", "prev_value_m", "value_m", "growth"]].copy()
        tp.columns = ["Player", "Position", "Age", "Value last season (€m)", "Value now (€m)", "Change (€m)"]
        st.dataframe(tp.round(2), hide_index=True, width="stretch")

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
               "Metrics only one provider publishes are labelled; they are empty for the other league.")
    td = tstats[(tstats.league == league) & (tstats.season == season)].copy()
    td = td.merge(ts[["league", "season", "club", "pts", "xpts", "club_skill", "squad_value_m", "is_reserve"]], on=["league", "season", "club"], how="left")
    avail = [c for c in STAT_LABELS if c in td.columns and td[c].notna().any()]
    view_for = st.multiselect("Statistics", avail, default=[c for c in ["possession", "xg_final", "shots", "shots_on_target", "big_chances", "touches_opp_box", "tackles"] if c in avail],
                              format_func=lambda c: STAT_LABELS[c], key="td_cols")
    side = st.radio("Show", ["For (the team)", "Against (the opponent)", "Both"], horizontal=True, key="td_side")
    cols = ["club", "matches"]
    for c in view_for:
        if side in ("For (the team)", "Both"):
            cols.append(c)
        if side in ("Against (the opponent)", "Both"):
            ac = "xga_final" if c == "xg_final" else f"{c}_against"
            if ac in td.columns:
                cols.append(ac)
    show = td[cols].copy()
    ren = {"club": "Club", "matches": "Games"}
    for c in cols[2:]:
        base_c = "xg_final" if c == "xga_final" else c.replace("_against", "")
        ren[c] = STAT_LABELS.get(base_c, base_c) + (" (against)" if c.endswith("_against") or c == "xga_final" else "")
    show = show.rename(columns=ren)
    st.dataframe(show.round(2).sort_values("Club"), hide_index=True, width="stretch")

    if "xg_final" in td.columns:
        st.markdown("#### Chance quality: xG created vs xG conceded (per game)")
        d = td.dropna(subset=["xg_final", "xga_final"]) if "xga_final" in td.columns else pd.DataFrame()
        if d.empty and "xg_final_against" in td.columns:
            d = td.rename(columns={"xg_final_against": "xga_final"}).dropna(subset=["xg_final", "xga_final"])
        if not d.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = [GREEN if v >= 0 else RED for v in d["club_skill"]]
            ax.scatter(d["xg_final"], d["xga_final"], c=colors, s=80, edgecolors="white")
            for _, r in d.iterrows():
                ax.annotate(short(r["club"], 14), (r["xg_final"], r["xga_final"]), fontsize=7.5, xytext=(4, 3), textcoords="offset points")
            ax.axvline(d["xg_final"].mean(), color="grey", lw=0.8, ls="--")
            ax.axhline(d["xga_final"].mean(), color="grey", lw=0.8, ls="--")
            ax.invert_yaxis()
            ax.set_xlabel("xG created per game", fontsize=9)
            ax.set_ylabel("xG conceded per game (axis inverted: up = better defence)", fontsize=9)
            ax.set_title("Green = above value trend, red = below (club skill)", fontsize=10)
            ax.grid(alpha=0.15)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.markdown("#### Rank the league on one statistic")
    if avail:
        metric = st.selectbox("Statistic", avail, format_func=lambda c: STAT_LABELS[c], key="td_metric")
        r = td[["club", metric]].dropna().sort_values(metric, ascending=False)
        fig, ax = plt.subplots(figsize=(9, max(4, len(r) * 0.34)))
        ax.barh(r["club"][::-1], r[metric][::-1], color=[BLUE if c != club else "#f1c40f" for c in r["club"][::-1]], edgecolor="white")
        ax.set_xlabel(STAT_LABELS[metric] + " per game", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown(f"#### {club}: this season against the previous one")
    cmp_rows = []
    order = SEASONS
    prev_season = order[order.index(season) + 1] if order.index(season) + 1 < len(order) else None
    if prev_season:
        a = tstats[(tstats.club == club) & (tstats.league == league) & (tstats.season == season)]
        b = tstats[(tstats.club == club) & (tstats.league == league) & (tstats.season == prev_season)]
        if len(a) and len(b):
            for c in avail:
                if pd.notna(a[c].iloc[0]) and pd.notna(b[c].iloc[0]):
                    cmp_rows.append([STAT_LABELS[c], b[c].iloc[0], a[c].iloc[0], a[c].iloc[0] - b[c].iloc[0]])
    if cmp_rows:
        cmp = pd.DataFrame(cmp_rows, columns=["Statistic", f"{prev_season}", f"{season}", "Change"])
        st.dataframe(cmp.round(2), hide_index=True, width="stretch")
    else:
        st.caption("No previous season in this league for this club.")
