    const dashboardState = { state: null, config: null, triggers: [], triggerSnapshots: [], selectedGameId: null, selectedSnapshot: null };
    const byId = (id) => document.getElementById(id);
    const pct = (value, digits=1) => value == null ? "—" : `${(Number(value) * 100).toFixed(digits)}%`;
    const money = (value) => value == null ? "—" : new Intl.NumberFormat("en-US", {style:"currency", currency:"USD", maximumFractionDigits:2}).format(value);
    const num = (value, digits=2) => value == null ? "—" : Number(value).toFixed(digits);
    const safe = (value) => String(value ?? "").replace(/[&<>'"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));
    const reliabilityChip = (flag) => `<span class="chip ${safe(flag || "unreliable")}">${safe(flag || "unknown")}</span>`;

    function statusChip(status) {
      const normalized = String(status || "MONITORING").toLowerCase().replace("_", "-");
      return `<span class="chip ${normalized}">${safe(status)}</span>`;
    }

    async function fetchJSON(url, options) {
      const response = await fetch(url, options);
      if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
      return response.json();
    }

    async function refresh() {
      try {
        const [state, config, triggers] = await Promise.all([
          fetchJSON("/api/state"), fetchJSON("/api/config"), fetchJSON("/api/triggers?limit=100")
        ]);
        dashboardState.state = state;
        dashboardState.config = config;
        dashboardState.triggers = triggers.triggers;
        dashboardState.triggerSnapshots = triggers.snapshots || [];
        if (!dashboardState.selectedGameId || !state.games.some(g => g.game_id === dashboardState.selectedGameId)) {
          const triggered = state.games.find(g => g.status === "TRIGGERED");
          dashboardState.selectedGameId = (triggered || state.games[0] || {}).game_id || null;
        }
        render();
      } catch (error) {
        byId("refreshState").textContent = "Dashboard API unavailable";
        byId("gameDetail").innerHTML = `<div class="fatal">${safe(error.message)}</div>`;
      }
    }

    function render() {
      renderMode(); renderConfig(); renderWatchList(); renderDetail(); renderTriggers();
      byId("refreshState").textContent = `Updated ${new Date().toLocaleTimeString()} · ${dashboardState.state.game_count} games`;
    }

    function renderMode() {
      const live = dashboardState.state.mode === "live";
      byId("modeBanner").className = `mode-banner${live ? " live" : ""}`;
      byId("modeBanner").textContent = live ? "LIVE MODE — public prices and paid CFBD feed" : "STUB MODE — replayed data, not live";
    }

    function renderConfig() {
      if (!dashboardState.config) return;
      for (const [key, value] of Object.entries(dashboardState.config)) {
        const input = document.querySelector(`[name="${key}"]`);
        if (input && document.activeElement !== input) input.value = value;
      }
    }

    function renderWatchList() {
      const games = [...dashboardState.state.games].sort((a,b) => (b.status === "TRIGGERED") - (a.status === "TRIGGERED"));
      if (!games.length) { byId("watchList").innerHTML = `<div class="empty">No games are currently on the watch list</div>`; return; }
      byId("watchList").innerHTML = games.map(game => {
        const state = game.state || {};
        const favRank = game.favorite_ap_rank ? `<span class="subtle">#${game.favorite_ap_rank}</span> ` : "";
        return `<button class="watch-game ${game.status === "TRIGGERED" ? "triggered" : ""} ${dashboardState.selectedGameId === game.game_id ? "selected" : ""}" data-game="${safe(game.game_id)}" type="button">
          <div class="watch-top"><span class="subtle">Q${safe(state.period || "—")} ${safe(state.clock || "—")}</span>${statusChip(game.status)}</div>
          <div class="team-row"><span class="team-name">${favRank}${safe(game.favorite)} <span class="favorite-mark">FAV</span></span><span class="score">${safe(game.favorite_score ?? "—")}</span></div>
          <div class="team-row"><span class="team-name">${safe(game.dog)}</span><span class="score">${safe(game.dog_score ?? "—")}</span></div>
          <div class="watch-top"><span class="subtle">Spread ${num(game.pregame_spread,1)}</span><span class="subtle">Deficit ${game.deficit == null ? "—" : safe(game.deficit)}</span></div>
        </button>`;
      }).join("");
      document.querySelectorAll("[data-game]").forEach(button => button.addEventListener("click", () => {
        dashboardState.selectedGameId = button.dataset.game; dashboardState.selectedSnapshot = null; renderWatchList(); renderDetail();
      }));
    }

    function renderDetail() {
      const game = dashboardState.selectedSnapshot || dashboardState.state.games.find(g => g.game_id === dashboardState.selectedGameId);
      if (!game) { byId("gameDetail").className = "empty"; byId("gameDetail").innerHTML = "Select a monitored game"; return; }
      byId("gameDetail").className = "";
      const state = game.state || {};
      byId("gameDetail").innerHTML = `
        <div class="section-heading"><h2>Game Detail</h2>${statusChip(game.status)}</div>
        <div class="scoreboard">
          <div class="score-team"><span class="subtle">Favorite · ${num(game.pregame_spread,1)}</span><strong>${safe(game.favorite)}</strong><span class="big-score">${safe(game.favorite_score ?? "—")}</span></div>
          <div class="game-clock"><span class="subtle">${safe(state.status || "not started")}</span><strong>Q${safe(state.period || "—")} ${safe(state.clock || "—")}</strong><span class="subtle">Possession: ${safe(state.possession || "unknown")}</span></div>
          <div class="score-team"><span class="subtle">Opponent</span><strong>${safe(game.dog)}</strong><span class="big-score">${safe(game.dog_score ?? "—")}</span></div>
        </div>
        <div class="detail-grid">
          ${enginePanel(game)}
          ${marketPanel(game)}
          ${gapPanel(game)}
          ${riskPanel(game)}
        </div>`;
    }

    function renderTriggers() {
      if (!dashboardState.triggers.length) { byId("triggerList").innerHTML = `<div class="empty">No triggers recorded</div>`; return; }
      byId("triggerList").innerHTML = dashboardState.triggers.map((row,index) => `<button class="trigger-row watch-game" data-trigger-index="${index}" type="button"><span class="subtle">${new Date(row.timestamp).toLocaleString()}</span><strong>${safe(row.favorite)} vs ${safe(row.dog)}</strong><span>D=${safe(row.threshold_crossed)}</span><span>${safe(row.fav_score)}-${safe(row.dog_score)}</span><span>Tier ${safe(row.tier_used || "—")}</span><span class="subtle">${safe(row.market_status || "NO_MARKET")}</span></button>`).join("");
      document.querySelectorAll("[data-trigger-index]").forEach(button => button.addEventListener("click", () => {
        const snapshot = dashboardState.triggerSnapshots[Number(button.dataset.triggerIndex)];
        if (snapshot) { dashboardState.selectedSnapshot = snapshot; dashboardState.selectedGameId = snapshot.game_id; renderWatchList(); renderDetail(); byId("detailSection").scrollIntoView({behavior:"smooth"}); }
      }));
    }

    byId("configToggle").addEventListener("click", () => byId("configPanel").classList.toggle("open"));
    byId("configForm").addEventListener("submit", async event => {
      event.preventDefault(); const payload = {};
      new FormData(event.target).forEach((value,key) => payload[key] = Number(value));
      try { dashboardState.config = await fetchJSON("/api/config", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)}); await refresh(); }
      catch (error) { alert(`Config not saved: ${error.message}`); }
    });
    refresh(); setInterval(refresh, 5000);
