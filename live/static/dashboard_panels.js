    function enginePanel(game) {
      const scoring = game.scoring;
      if (!scoring) return `<article class="panel"><div class="panel-head"><h3>The Engine's Read</h3><span class="chip unreliable">waiting</span></div><div class="empty">No trigger estimate yet</div></article>`;
      const finalWin = scoring.tier_1.favorite_final_win;
      const erased = scoring.tier_1.deficit_erased;
      const tier3 = scoring.tier_3;
      const conditional = findEstimate(scoring, "conditional_rate_full", "favorite_final_win");
      const ranking = findEstimate(scoring, "ranking_rate", "favorite_final_win");
      const historicalMarket = findEstimate(scoring, "market_no_vig_historical", "favorite_final_win");
      const disagreement = tier3 ? Math.abs(tier3.calibrated_prob - erased.value) : 0;
      return `<article class="panel">
        <div class="panel-head"><h3>The Engine's Read</h3><span class="chip tier" title="${safe(scoring.tier_reasons[`tier_${scoring.tier_used}`])}">Tier ${scoring.tier_used}</span></div>
        <div class="primary-prob"><strong>${pct(finalWin.value)}</strong><span>Favorite final-win probability</span></div>
        <div class="metric-row"><span class="metric-label">Evidence</span><span class="metric-value">${safe(finalWin.n_events)} events · ${safe(finalWin.n_games)} games · ${safe(finalWin.n_seasons)} seasons</span></div>
        <div class="metric-row"><span class="metric-label">Reliability</span>${reliabilityChip(finalWin.reliability_flag)}</div>
        <div class="metric-row"><span class="metric-label">Probability deficit is erased</span><span class="metric-value">${pct(erased.value)}</span></div>
        <div class="warning amber">These are different outcomes: erasing the deficit does not mean winning the game.</div>
        <div class="context-block"><h4>Historical descriptive · Tier 2</h4>
          ${estimateRow("Conditional final-win rate", conditional)}
          ${estimateRow("AP-ranking final-win rate", ranking)}
          ${estimateRow("What pregame markets historically charged", historicalMarket)}
        </div>
        <div class="context-block"><h4>Tier 3 · deficit-erased model</h4>
          ${tier3 ? conformalBand(tier3) : `<div class="warning amber">${safe(scoring.tier_3_unavailable_reason || "Tier 3 unavailable")}</div>`}
          ${tier3 && disagreement > .10 ? `<div class="warning">LABEL-MATCHED DISAGREEMENT: Tier 1 deficit-erased ${pct(erased.value)} vs Tier 3 ${pct(tier3.calibrated_prob)}.</div>` : ""}
        </div>
      </article>`;
    }

    function findEstimate(scoring, metric, label) {
      return (scoring.tier_2?.[metric] || []).find(row => row.label === label) || null;
    }

    function estimateRow(label, row) {
      if (!row) return `<div class="metric-row"><span class="metric-label">${safe(label)}</span><span class="subtle">unavailable</span></div>`;
      return `<div class="metric-row"><span class="metric-label">${safe(label)}</span><span class="metric-value">${pct(row.value)} · n=${safe(row.n_events)} ${reliabilityChip(row.reliability_flag)}</span></div>`;
    }

    function conformalBand(tier3) {
      const left = Math.max(0, tier3.conformal_lower) * 100;
      const width = Math.max(0, tier3.conformal_upper - tier3.conformal_lower) * 100;
      const point = tier3.calibrated_prob * 100;
      return `<div class="metric-row"><span class="metric-label">Probability deficit is erased</span><span class="metric-value">${pct(tier3.calibrated_prob)}</span></div>
        <div class="band-wrap"><div class="band-scale" aria-label="Conformal interval from ${pct(tier3.conformal_lower)} to ${pct(tier3.conformal_upper)}">
          <div class="band-range" style="left:${left}%;width:${width}%"></div><div class="band-point" style="left:${point}%"></div>
        </div><div class="band-labels"><span>${pct(tier3.conformal_lower)}</span><span>95% split-conformal band · q-hat ${num(tier3.conformal_q_hat,3)}</span><span>${pct(tier3.conformal_upper)}</span></div></div>`;
    }

    function marketPanel(game) {
      const market = game.market || {status:"NO_MARKET", quotes:{}, errors:{}};
      const venues = ["kalshi", "polymarket"];
      return `<article class="panel"><div class="panel-head"><h3>Market Lines</h3>${statusChip(market.status)}</div>
        <div class="market-grid">${venues.map(venue => {
          const quote = market.quotes?.[venue]; const error = market.errors?.[venue];
          if (!quote) return `<div class="market-venue"><div class="market-title"><span>${venue}</span><span class="chip no-market">${error ? "error" : "no market"}</span></div><div class="subtle">${safe(error || "No confidently mapped market")}</div></div>`;
          const depth = (quote.depth_top_levels || []).reduce((sum,row) => sum + Number(row.ask_size || 0), 0);
          return `<div class="market-venue ${market.best_venue === venue ? "best" : ""}"><div class="market-title"><span>${venue}</span>${market.best_venue === venue ? '<span class="chip ok">best gap</span>' : ""}</div>
            <div class="metric-row"><span>Bid / ask / mid</span><strong>${pct(quote.best_bid)} / ${pct(quote.best_ask)} / ${pct(quote.mid)}</strong></div>
            <div class="metric-row"><span>Raw · what you pay</span><strong>${pct(quote.implied_prob_raw)}</strong></div>
            <div class="metric-row"><span>No-vig · market belief</span><strong>${pct(quote.implied_prob_no_vig)}</strong></div>
            <div class="metric-row"><span>Spread</span><strong>${pct(quote.spread,2)}</strong></div>
            <div class="metric-row"><span>Top ask depth</span><strong>${num(depth,1)}</strong></div>
            <div class="metric-row"><span>Freshness</span><span class="chip ${quote.is_stale ? "error" : "ok"}">${quote.is_stale ? "stale" : "fresh"}</span></div>
          </div>`;
        }).join("")}</div></article>`;
    }

    function gapPanel(game) {
      const venues = game.risk?.venues || {};
      const reads = Object.entries(venues).filter(([,read]) => read.status === "OK");
      return `<article class="panel"><div class="panel-head"><h3>The Gap</h3><span class="subtle">Tier 1 final-win − live no-vig</span></div>
        ${reads.length ? reads.map(([venue,read]) => `<div class="context-block"><div class="market-title"><span>${safe(venue)}</span><span class="gap-number ${read.gap_no_vig > 0 ? "positive" : "negative"}">${read.gap_no_vig >= 0 ? "+" : ""}${pct(read.gap_no_vig)}</span></div>
          <div class="metric-row"><span>Bid-ask spread</span><strong>${pct(read.spread,2)}</strong></div>
          ${read.survives_friction ? '<div class="chip reliable">gap exceeds spread</div>' : '<div class="warning">DOES NOT SURVIVE FRICTION — spread is as large as or larger than the gap.</div>'}
        </div>`).join("") : `<div class="empty">No fresh market quote; engine estimate remains available</div>`}
      </article>`;
    }

    function riskPanel(game) {
      const risk = game.risk || {}; const venues = risk.venues || {};
      const preferred = game.market?.best_venue;
      const entry = (preferred && venues[preferred]?.status === "OK") ? [preferred, venues[preferred]] : Object.entries(venues).find(([,read]) => read.status === "OK");
      if (!entry) return `<article class="panel risk-panel"><div class="panel-head"><h3>Risk & Variance</h3><span class="chip no-market">no price</span></div><div class="warning amber">${safe(risk.notice || "A real offered price is required for EV and sizing.")}</div></article>`;
      const [venue, read] = entry;
      const lossMajority = read.engine_probability < .5 ? "At this estimate, losses are more likely than wins even if the edge is real." : "Losing streaks remain plausible even with a win probability above 50%.";
      return `<article class="panel risk-panel"><div class="panel-head"><h3>Risk & Variance · ${safe(venue)}</h3><span class="chip ${read.positive_ev ? "reliable" : "error"}">${read.positive_ev ? "positive point EV" : "no edge"}</span></div>
        <div class="warning amber">${safe(risk.notice)}</div>
        ${!read.positive_ev ? '<div class="warning">DO NOT BET — expected value is zero or negative at the raw offered price.</div>' : ""}
        ${read.favorite_longshot_bias_note ? '<div class="warning amber">EXTREME-LOW PRICE: prediction markets exhibit favorite-longshot bias near price extremes; true win probability may be below the quoted implied probability. Informational only.</div>' : ""}
        <div class="risk-grid">
          <div class="risk-cell"><h4>Expected Value</h4><div class="risk-hero ${read.ev_per_dollar > 0 ? "positive" : "negative"}">${read.ev_per_dollar >= 0 ? "+" : ""}${pct(read.ev_per_dollar)}</div><p class="subtle">Per dollar at the raw ${pct(1/read.decimal_odds)} offer. No conformal lower-bound EV exists for this label.</p></div>
          <div class="risk-cell"><h4>Sizing Suggestion</h4><div class="risk-hero">${money(read.suggested_dollars)}</div><p>${pct(read.suggested_fraction)} of bankroll · full Kelly ${pct(read.full_kelly_fraction)}</p><p class="subtle" title="${safe(read.sizing_formula)}">${pct(dashboardState.config?.kelly_fraction ?? 0.25, 0)} Kelly × reliability ${num(read.reliability_factor,2)}${read.cap_applied ? " · cap applied" : ""}. Human decision only.</p></div>
          <div class="risk-cell"><h4>Streak Reality · ${safe(read.season_bets)} bets</h4><table class="streak-table">${[3,5,7,10].map(n => `<tr><td>${n} losses</td><td>${pct(read.streaks[n].probability)} · ${num(read.streaks[n].expected_count,2)} expected</td></tr>`).join("")}</table><p class="subtle">${lossMajority}</p></div>
          <div class="risk-cell"><h4>Drawdown-Floor Risk</h4><div class="risk-hero ${read.ruin_warning ? "negative" : "positive"}">${pct(read.ruin_probability)}</div><p>Chance of touching ${pct(read.drawdown_floor)} of starting bankroll.</p>${read.ruin_warning ? `<p class="warning">Above ${pct(read.ruin_comfort_threshold)} comfort. Reduce to ${money(read.comfort_dollars)} (${pct(read.comfort_fraction)}).</p>` : '<p class="subtle">Within configured comfort threshold.</p>'}</div>
        </div></article>`;
    }
