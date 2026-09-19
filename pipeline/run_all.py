"""Rebuilds every processed table from the raw scrape (data/raw/)."""
import build_matches
import build_skill
import estimate_xg

if __name__ == "__main__":
    print("== 1/3 team-match table ==")
    build_matches.main()
    print("== 2/3 xG estimation + xPts ==")
    estimate_xg.main()
    print("== 3/3 club skill, manager skill, organic growth ==")
    build_skill.main()
