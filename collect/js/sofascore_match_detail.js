// Run inside a sofascore.com tab (browser console / javascript_tool), AFTER window.__matchList has
// been set to an array of {league, season, id} (Sofascore event ids, one entry per finished match).
//
// For every match, collects:
//   - lineups          -> player roster, position, starter/sub, and each player's aggregate match
//                          statistics (already embedded in the lineups response, no extra call)
//   - shotmap          -> every shot in the match, with coordinates, xG/xGOT and the scoring player
//   - graph            -> match momentum (minute-by-minute value)
//   - average-positions-> each player's average (x, y) on the pitch
//   - per player (minutesPlayed > 0 only): heatmap (touch points) and rating-breakdown
//     (event-level passes / dribbles / defensive actions / ball-carries, each with coordinates
//     and outcome -- this is the data behind Sofascore's own Shot/Pass/Drib/Def pitch maps)
//
// Long-running: keeps going past the 45s tool timeout, state lives in window.__md. Call
// window.__md.progress() any time to check on it; pull window.__md.matches (an object keyed by
// match id) to read out finished matches, and window.__md.flushed to see which ids were already
// pulled and can be dropped to save memory.
//
// Concurrency 6 (Sofascore's own site issues about that many parallel requests), 3 retries per call.

window.__md = { matches: {}, errors: [], startedAt: Date.now(), doneIds: new Set() };

(async () => {
  const getj = async (u) => {
    for (let a = 0; a < 3; a++) {
      const r = await fetch(u);
      if (r.status === 200) return await r.json();
      if (r.status === 404) return null;
      await new Promise((res) => setTimeout(res, 1000 * (a + 1)));
    }
    return undefined; // undefined = gave up after retries; null = confirmed 404
  };

  async function pool(items, worker, concurrency = 6) {
    let i = 0;
    async function run() {
      while (i < items.length) {
        const idx = i++;
        try {
          await worker(items[idx], idx);
        } catch (e) {
          window.__md.errors.push(String(e));
        }
      }
    }
    await Promise.all(Array.from({ length: concurrency }, run));
  }

  window.__md.progress = () => ({
    total: window.__matchList.length,
    done: window.__md.doneIds.size,
    inMemory: Object.keys(window.__md.matches).length,
    errors: window.__md.errors.length,
    elapsed_s: Math.round((Date.now() - window.__md.startedAt) / 1000),
  });

  await pool(window.__matchList, async (m) => {
    const id = m.id;
    const [lineups, shotmap, graph, avgpos] = await Promise.all([
      getj(`https://api.sofascore.com/api/v1/event/${id}/lineups`),
      getj(`https://api.sofascore.com/api/v1/event/${id}/shotmap`),
      getj(`https://api.sofascore.com/api/v1/event/${id}/graph`),
      getj(`https://api.sofascore.com/api/v1/event/${id}/average-positions`),
    ]);
    if (!lineups) { window.__md.errors.push(`no lineups ${id}`); window.__md.doneIds.add(id); return; }

    const players = [];
    for (const side of ['home', 'away']) {
      for (const p of (lineups[side] || {}).players || []) {
        const mins = (p.statistics || {}).minutesPlayed || 0;
        if (mins <= 0) continue;
        players.push({ id: p.player.id, name: p.player.name, position: p.position, isHome: side === 'home',
                       substitute: !!p.substitute, statistics: p.statistics });
      }
    }

    await pool(players, async (pl) => {
      const [heatmap, rb] = await Promise.all([
        getj(`https://api.sofascore.com/api/v1/event/${id}/player/${pl.id}/heatmap`),
        getj(`https://api.sofascore.com/api/v1/event/${id}/player/${pl.id}/rating-breakdown`),
      ]);
      pl.heatmap = heatmap ? heatmap.heatmap : null;
      pl.events = rb || null;
    }, 6);

    window.__md.matches[id] = {
      league: m.league, season: m.season, round: m.round, id, ts: m.ts, h: m.h, a: m.a, hs: m.hs, as: m.as,
      avgpos: avgpos || null,
      momentum: graph ? graph.graphPoints : null,
      shots: shotmap ? shotmap.shotmap : null,
      players,
    };
    window.__md.doneIds.add(id);
  }, 6);

  window.__md.finished = true;
})();
'started';
