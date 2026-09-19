// Run inside a sofascore.com tab (browser console / javascript_tool).
// Step 1: fixtures for the Eerste Divisie (2022/23 - 2026/27) -> window.__matches.rows
// Step 2: team match stats for every finished match -> window.__ss_stats
window.__matches = {done: false, rows: [], log: []};
window.__ss_stats = {done: false, rows: [], log: []};
(async () => {
  const getj = async (u) => {
    for (let a = 0; a < 3; a++) {
      const r = await fetch(u);
      if (r.status === 200) return await r.json();
      if (r.status === 404) return {events: []};
      await new Promise((res) => setTimeout(res, 1500 * (a + 1)));
    }
    return null;
  };
  const seasons = {'22/23': 42258, '23/24': 52556, '24/25': 61667, '25/26': 77156, '26/27': 96187};
  for (const [season, sid] of Object.entries(seasons)) {
    for (let i = 1; i <= 38; i += 6) {
      const rounds = [i, i + 1, i + 2, i + 3, i + 4, i + 5].filter((r) => r <= 38);
      const res = await Promise.all(rounds.map((rd) =>
        getj(`https://api.sofascore.com/api/v1/unique-tournament/131/season/${sid}/events/round/${rd}`).then((j) => ({rd, j}))));
      for (const {rd, j} of res) {
        if (!j) window.__matches.log.push(`FAILED ${season} r${rd}`);
        for (const e of ((j || {}).events || [])) {
          window.__matches.rows.push({league: 'Eerste Divisie', season, round: rd, event_id: e.id, ts: e.startTimestamp,
            home_id: e.homeTeam.id, home: e.homeTeam.name, away_id: e.awayTeam.id, away: e.awayTeam.name,
            hg: e.homeScore ? e.homeScore.current : null, ag: e.awayScore ? e.awayScore.current : null,
            finished: e.status.type === 'finished' ? 1 : 0});
        }
      }
    }
  }
  window.__matches.done = true;

  const KEYS = ['ballPossession', 'expectedGoals', 'expectedGoalsOnTarget', 'bigChanceCreated', 'totalShotsOnGoal',
    'shotsOnGoal', 'shotsOffGoal', 'blockedScoringAttempt', 'totalShotsInsideBox', 'totalShotsOutsideBox',
    'bigChanceScored', 'bigChanceMissed', 'hitWoodwork', 'cornerKicks', 'fouls', 'offsides', 'yellowCards',
    'redCards', 'passes', 'accuratePasses', 'accurateLongBalls', 'accurateCross', 'throwIns', 'finalThirdEntries',
    'touchesInOppBox', 'dispossessed', 'duelWonPercent', 'aerialDuelsPercentage', 'groundDuelsPercentage',
    'dribblesPercentage', 'wonTacklePercent', 'totalTackle', 'interceptionWon', 'ballRecovery', 'totalClearance',
    'goalkeeperSaves', 'goalsPrevented', 'errorsLeadToShot', 'errorsLeadToGoal', 'highClaims',
    'kilometersCovered', 'numberOfSprints', 'accurateThroughBall', 'fouledFinalThird'];
  window.__ss_stats.keys = KEYS;
  const todo = window.__matches.rows.filter((r) => r.finished).map((r) => r.event_id);
  for (let i = 0; i < todo.length; i += 6) {
    await Promise.all(todo.slice(i, i + 6).map(async (id) => {
      const st = await getj(`https://api.sofascore.com/api/v1/event/${id}/statistics`);
      if (!st) { window.__ss_stats.log.push('FAILED ' + id); return; }
      const all = (st.statistics || []).find((p) => p.period === 'ALL');
      const by = {};
      if (all) for (const g of (all.groups || [])) for (const it of (g.statisticsItems || [])) by[it.key] = it;
      const v = [];
      for (const k of KEYS) {
        const it = by[k];
        v.push(it && it.homeValue !== undefined ? Number(it.homeValue) : null, it && it.awayValue !== undefined ? Number(it.awayValue) : null);
      }
      window.__ss_stats.rows.push({event_id: id, has_stats: all ? 1 : 0, v});
    }));
  }
  window.__ss_stats.done = true;
})();
