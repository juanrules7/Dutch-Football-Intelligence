// Run inside a fotmob.com tab (browser console / javascript_tool).
// Collects every regular-season Eredivisie match (2022/23 - 2026/27) with its
// team match stats. Result lands in window.__fm_ere = {done, matches, stats}.
window.__fm_ere = {done: false, matches: [], stats: [], log: []};
(async () => {
  const getj = async (u) => {
    for (let a = 0; a < 3; a++) {
      const r = await fetch(u);
      if (r.status === 200) return await r.json();
      await new Promise((res) => setTimeout(res, 1500 * (a + 1)));
    }
    return null;
  };
  const num = (x) => {
    if (x === null || x === undefined) return null;
    const m = String(x).match(/-?\d+(\.\d+)?/);
    return m ? parseFloat(m[0]) : null;
  };
  const KEYS = ['BallPossesion', 'expected_goals', 'expected_goals_non_penalty', 'expected_goals_on_target',
    'expected_goals_open_play', 'expected_goals_set_play', 'total_shots', 'ShotsOnTarget', 'ShotsOffTarget',
    'blocked_shots', 'shots_inside_box', 'shots_outside_box', 'shots_woodwork', 'big_chance',
    'big_chance_missed_title', 'corners', 'Offsides', 'fouls', 'yellow_cards', 'red_cards', 'passes',
    'accurate_passes', 'long_balls_accurate', 'accurate_crosses', 'player_throws', 'touches_opp_box',
    'own_half_passes', 'opposition_half_passes', 'duel_won', 'ground_duels_won', 'aerials_won',
    'dribbles_succeeded', 'matchstats.headers.tackles', 'interceptions', 'clearances', 'keeper_saves'];
  window.__fm_ere.keys = KEYS;
  const seasons = ['2022/2023', '2023/2024', '2024/2025', '2025/2026', '2026/2027'];
  for (const season of seasons) {
    const fx = await getj(`https://www.fotmob.com/api/data/leagues?id=57&season=${encodeURIComponent(season)}&tab=fixtures`);
    const all = ((fx || {}).fixtures || {}).allMatches || [];
    const todo = [];
    for (const m of all) {
      const isRegular = /^\d+$/.test(String(m.round));
      const finished = !!(m.status && m.status.finished);
      const score = (m.status && m.status.scoreStr) ? m.status.scoreStr.split(' - ') : [null, null];
      window.__fm_ere.matches.push({league: 'Eredivisie', season, round: m.round, regular: isRegular ? 1 : 0, match_id: m.id,
        utc: m.status.utcTime, home_id: m.home.id, home: m.home.name, away_id: m.away.id, away: m.away.name,
        hg: score[0] === null ? null : parseInt(score[0]), ag: score[1] === null ? null : parseInt(score[1]),
        finished: finished ? 1 : 0});
      if (isRegular && finished) todo.push(m.id);
    }
    for (let i = 0; i < todo.length; i += 6) {
      await Promise.all(todo.slice(i, i + 6).map(async (id) => {
        const d = await getj(`https://www.fotmob.com/api/data/matchDetails?matchId=${id}`);
        if (!d) { window.__fm_ere.log.push('FAILED ' + id); return; }
        const groups = ((((d.content || {}).stats || {}).Periods || {}).All || {}).stats || [];
        const by = {};
        for (const g of groups) for (const it of (g.stats || [])) {
          if (it.stats && (it.stats[0] !== null || it.stats[1] !== null)) by[it.key] = it.stats;
        }
        const v = [];
        for (const k of KEYS) { const s = by[k] || [null, null]; v.push(num(s[0]), num(s[1])); }
        window.__fm_ere.stats.push({match_id: id, v});
      }));
    }
  }
  window.__fm_ere.done = true;
})();
