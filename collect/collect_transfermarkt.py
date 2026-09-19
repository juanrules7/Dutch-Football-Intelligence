"""
Collects Transfermarkt data for the Eredivisie (NL1) and Eerste Divisie (NL2):

  tm_clubs.csv          one row per league x season x club, with total squad value
  tm_player_values.csv  one row per league x season x club x player, with market value
  tm_managers.csv       one row per club x head-coach stint (appointed / until / matches)

Season convention: `season` is the START year (2025 = 2025/26).
Player values are collected from 2021 so that organic growth can be computed
for 2022/23 (needs the previous season's value).

The script is resumable: rows already saved are skipped on the next run.

Usage: python collect/collect_transfermarkt.py
"""
import os
import re
import time
import urllib.request

import pandas as pd
from bs4 import BeautifulSoup

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "..", "data", "raw")
os.makedirs(RAW, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.transfermarkt.com/",
}
DELAY = 2.0

LEAGUES = {
    "Eredivisie": ("eredivisie", "NL1"),
    "Eerste Divisie": ("eerste-divisie", "NL2"),
}
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]


def get(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            print(f"    retry {attempt + 1}/{retries} for {url}: {e}")
            time.sleep(8 * (attempt + 1))
    return None


def parse_value(text):
    text = text.replace("€", "").replace(",", "").strip()
    text = text.encode("ascii", "ignore").decode()
    try:
        if text.endswith("bn"):
            return float(text[:-2]) * 1000
        if text.endswith("m"):
            return float(text[:-1])
        if text.endswith("k"):
            return float(text[:-1]) / 1000
    except ValueError:
        return None
    return None


def collect_clubs():
    path = os.path.join(RAW, "tm_clubs.csv")
    done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    rows = done.to_dict("records") if len(done) else []
    have = {(r["league"], r["season"]) for r in rows}
    for league, (slug, code) in LEAGUES.items():
        for season in SEASONS:
            if (league, season) in have:
                continue
            url = f"https://www.transfermarkt.com/{slug}/startseite/wettbewerb/{code}/plus/?saison_id={season}"
            html = get(url)
            time.sleep(DELAY)
            if not html:
                continue
            table = BeautifulSoup(html, "html.parser").find("table", {"class": "items"})
            if not table:
                continue
            n = 0
            for tr in table.find("tbody").find_all("tr", recursive=False):
                tds = tr.find_all("td", recursive=False)
                if len(tds) < 7:
                    continue
                a = tds[1].find("a", href=re.compile(r"/startseite/verein/\d+"))
                if not a:
                    continue
                m = re.search(r"/([^/]+)/startseite/verein/(\d+)", a["href"])
                rows.append({
                    "league": league, "season": season, "tm_name": tds[1].get_text(strip=True),
                    "slug": m.group(1), "team_id": int(m.group(2)),
                    "squad_size": tds[2].get_text(strip=True), "avg_age": tds[3].get_text(strip=True),
                    "squad_value_m": parse_value(tds[6].get_text(strip=True)),
                })
                n += 1
            print(f"clubs {league} {season}: {n}")
            pd.DataFrame(rows).to_csv(path, index=False)
    return pd.DataFrame(rows)


def collect_player_values(clubs):
    path = os.path.join(RAW, "tm_player_values.csv")
    done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    rows = done.to_dict("records") if len(done) else []
    have = {(r["season"], r["team_id"]) for r in rows}
    for _, c in clubs.iterrows():
        key = (c["season"], c["team_id"])
        if key in have:
            continue
        url = (f"https://www.transfermarkt.com/{c['slug']}/kader/verein/{c['team_id']}"
               f"/saison_id/{c['season']}/plus/1")
        html = get(url)
        time.sleep(DELAY)
        if not html:
            continue
        table = BeautifulSoup(html, "html.parser").find("table", {"class": "items"})
        n = 0
        if table:
            for tr in table.find("tbody").find_all("tr", recursive=False):
                tds = tr.find_all("td", recursive=False)
                if len(tds) < 5:
                    continue
                a = tr.find("a", href=re.compile(r"/profil/spieler/\d+"))
                if not a:
                    continue
                pid = re.search(r"/spieler/(\d+)", a["href"]).group(1)
                value = parse_value(tds[-1].get_text(strip=True))
                if value is None:
                    continue
                inline_rows = tds[1].find_all("tr")
                position = inline_rows[1].get_text(strip=True) if len(inline_rows) > 1 else ""
                age_m = re.search(r"\((\d+)\)", tds[2].get_text(strip=True))
                rows.append({
                    "league": c["league"], "season": c["season"], "team_id": c["team_id"],
                    "team": c["tm_name"], "player_id": pid, "player_name": a.get_text(strip=True),
                    "position": position, "age": int(age_m.group(1)) if age_m else None,
                    "value_m": value,
                })
                n += 1
        print(f"players {c['league']} {c['season']} {c['tm_name']}: {n}")
        pd.DataFrame(rows).to_csv(path, index=False)
        have.add(key)


def collect_managers(clubs):
    path = os.path.join(RAW, "tm_managers.csv")
    done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    rows = done.to_dict("records") if len(done) else []
    have = {r["team_id"] for r in rows}
    uniq = clubs[clubs["season"] >= 2022].drop_duplicates("team_id")
    for _, c in uniq.iterrows():
        if c["team_id"] in have:
            continue
        url = (f"https://www.transfermarkt.com/{c['slug']}/mitarbeiterhistorie/verein/"
               f"{c['team_id']}/personalie_id/1")
        html = get(url)
        time.sleep(DELAY)
        if not html:
            continue
        table = BeautifulSoup(html, "html.parser").find("table", {"class": "items"})
        n = 0
        if table:
            for tr in table.find("tbody").find_all("tr", recursive=False):
                tds = tr.find_all("td", recursive=False)
                if len(tds) < 7:
                    continue
                # the first link in the row wraps the photo, so it has no text
                anchors = tr.find_all("a", href=re.compile(r"/profil/trainer/\d+"))
                a = next((x for x in anchors if x.get_text(strip=True)), anchors[0] if anchors else None)
                name = a.get_text(strip=True) if a else tds[0].get_text(" ", strip=True)
                mid = re.search(r"/trainer/(\d+)", a["href"]).group(1) if a else None
                rows.append({
                    "team_id": c["team_id"], "team": c["tm_name"], "manager": name, "manager_id": mid,
                    "date_from": tds[2].get_text(strip=True), "date_to": tds[3].get_text(strip=True),
                    "matches": tds[5].get_text(strip=True), "ppg": tds[6].get_text(strip=True),
                })
                n += 1
        print(f"managers {c['tm_name']}: {n}")
        pd.DataFrame(rows).to_csv(path, index=False)
        have.add(c["team_id"])


if __name__ == "__main__":
    import sys

    clubs_df = collect_clubs()
    if sys.argv[1:] == ["managers"]:
        collect_managers(clubs_df)
    else:
        collect_managers(clubs_df)
        collect_player_values(clubs_df)
    print("done")
