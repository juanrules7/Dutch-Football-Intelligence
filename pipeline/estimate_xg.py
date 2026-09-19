"""
Fills the xG gap in the Eerste Divisie (no provider publishes xG there before
2025/26) and computes Poisson expected points for every match.

Estimator: a non-negative linear model on shot counts, trained on the matches
where real xG exists. It deliberately uses only inputs that do not depend on
whether the shots went in (shots by zone, shots on target, blocked shots, big
chances created, corners, woodwork) - goals, big chances scored/missed and the
result never enter, otherwise finishing luck would leak into the "expected"
number and hide exactly what xPts is meant to expose.

Validation is done where the truth is known: the Eredivisie. The model is
trained on ONE season and asked to reproduce xG in the OTHER seasons, which is
the same situation as Eerste Divisie 2025/26 -> 2022/23-2024/25.

Outputs
  data/processed/team_matches.csv      team_matches_raw + xg_final / xga_final / xg_source / xpts
  data/processed/xg_model_report.json  validation numbers shown in the app
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_val_predict

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "..", "data", "processed")

FEATURES = ["shots_inside_box", "shots_outside_box", "shots_on_target", "shots_blocked",
            "big_chances", "corners", "woodwork"]
GOALS = np.arange(9)


def poisson_xpts(xgf, xga):
    """Expected points from two Poisson goal distributions (goals 0..8)."""
    xgf = np.maximum(np.asarray(xgf, dtype=float), 1e-9)
    xga = np.maximum(np.asarray(xga, dtype=float), 1e-9)
    pf = stats.poisson.pmf(GOALS[None, :], xgf[:, None])
    pa = stats.poisson.pmf(GOALS[None, :], xga[:, None])
    mat = pf[:, :, None] * pa[:, None, :]
    idx = np.arange(len(GOALS))
    win = (mat * (idx[:, None] > idx[None, :])[None, :, :]).sum(axis=(1, 2))
    draw = (mat * (idx[:, None] == idx[None, :])[None, :, :]).sum(axis=(1, 2))
    return 3 * win + draw


def make_model():
    return LinearRegression(positive=True, fit_intercept=False)


def prep(df):
    # Sofascore leaves an event count out of the payload when it is zero
    # (e.g. a match with no big chances), so a gap next to a present shot total means 0.
    return df[FEATURES].fillna(0)


def season_xpts(df, xg_col, xga_col):
    d = df.copy()
    d["xpts"] = poisson_xpts(d[xg_col], d[xga_col])
    return d.groupby(["season", "team"])["xpts"].sum()


def main():
    tm = pd.read_csv(os.path.join(OUT, "team_matches_raw.csv"), parse_dates=["date"])
    report = {"features": FEATURES}

    # ---- 1. In-sample-league validation on the Eredivisie (truth known everywhere) ----
    ere = tm[(tm.league == "Eredivisie") & tm.xg.notna() & tm.shots_inside_box.notna()].copy()
    rows = []
    for train_season in ["22/23", "23/24", "24/25", "25/26"]:
        train = ere[ere.season == train_season]
        model = make_model().fit(prep(train), train["xg"])
        for test_season in sorted(ere.season.unique()):
            if test_season == train_season or test_season == "26/27":
                continue
            test = ere[ere.season == test_season].copy()
            test["xg_hat"] = model.predict(prep(test))
            r2 = 1 - ((test.xg - test.xg_hat) ** 2).sum() / ((test.xg - test.xg.mean()) ** 2).sum()
            mae = (test.xg - test.xg_hat).abs().mean()
            # what matters for manager skill: season-level xPts built from estimated vs real xG
            opp = test[["src_id", "team", "xg_hat"]].rename(columns={"team": "opponent", "xg_hat": "xga_hat"})
            test = test.merge(opp, on=["src_id", "opponent"], how="left")
            real = season_xpts(test, "xg", "xg_against")
            est = season_xpts(test, "xg_hat", "xga_hat")
            rows.append({"train": train_season, "test": test_season, "r2_match": r2, "mae_match": mae,
                         "season_xpts_corr": float(np.corrcoef(real, est)[0, 1]),
                         "season_xpts_mae": float((real - est).abs().mean()),
                         "season_xpts_sd_real": float(real.std())})
    val = pd.DataFrame(rows)
    report["cross_season_validation_eredivisie"] = val.round(3).to_dict("records")
    report["cross_season_summary"] = {k: round(float(val[k].mean()), 3) for k in
                                      ["r2_match", "mae_match", "season_xpts_corr", "season_xpts_mae", "season_xpts_sd_real"]}

    # ---- 2. Train the real estimator on Eerste Divisie matches that have real xG ----
    eer = tm[tm.league == "Eerste Divisie"].copy()
    train = eer[eer.xg.notna() & eer.shots_inside_box.notna()]
    X, y = prep(train), train["xg"]
    cv = GroupKFold(n_splits=5)
    pred = cross_val_predict(make_model(), X, y, cv=cv, groups=train["src_id"])
    report["eerste_divisie_training"] = {
        "matches_used": int(train.src_id.nunique()), "team_rows": int(len(train)),
        "cv_r2": float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        "cv_mae": float((y - pred).abs().mean()),
        "mean_real_xg": float(y.mean()),
    }
    # season-level check inside the Eerste Divisie itself (out-of-fold predictions, 2025/26)
    chk = train.assign(xg_hat=pred)
    chk = chk[chk.season == "25/26"]
    opp = chk[["src_id", "team", "xg_hat"]].rename(columns={"team": "opponent", "xg_hat": "xga_hat"})
    chk = chk.merge(opp, on=["src_id", "opponent"], how="left")
    real_xp = season_xpts(chk, "xg", "xg_against")
    est_xp = season_xpts(chk, "xg_hat", "xga_hat")
    report["eerste_divisie_training"].update({
        "season_xpts_corr": float(np.corrcoef(real_xp, est_xp)[0, 1]),
        "season_xpts_mae": float((real_xp - est_xp).abs().mean()),
        "season_xpts_sd_real": float(real_xp.std()),
    })

    model = make_model().fit(X, y)
    report["eerste_divisie_coefficients"] = {f: round(float(c), 4) for f, c in zip(FEATURES, model.coef_)}

    # ---- 3. Apply: real xG where it exists, estimated otherwise ----
    tm["xg_est"] = np.nan
    has_feat = tm.shots_inside_box.notna()
    est_mask = has_feat & (tm.league == "Eerste Divisie")
    tm.loc[est_mask, "xg_est"] = model.predict(prep(tm[est_mask]))
    tm["xg_final"] = tm["xg"].where(tm["xg"].notna(), tm["xg_est"])
    tm["xg_source"] = np.where(tm["xg"].notna(), "real", np.where(tm["xg_final"].notna(), "estimated", "missing"))
    opp = tm[["src_id", "team", "xg_final"]].rename(columns={"team": "opponent", "xg_final": "xga_final"})
    tm = tm.merge(opp, on=["src_id", "opponent"], how="left")
    tm["xpts"] = np.where(tm.xg_final.notna() & tm.xga_final.notna(), poisson_xpts(tm.xg_final.fillna(1), tm.xga_final.fillna(1)), np.nan)

    missing = tm[tm.xg_source == "missing"]
    report["matches_without_any_xg"] = int(missing.src_id.nunique())
    tm.to_csv(os.path.join(OUT, "team_matches.csv"), index=False)
    with open(os.path.join(OUT, "xg_model_report.json"), "w") as f:
        json.dump(report, f, indent=1)

    print(json.dumps({k: v for k, v in report.items() if k != "cross_season_validation_eredivisie"}, indent=1))
    print(val.round(3).to_string())
    print(tm.groupby(["league", "season", "xg_source"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
