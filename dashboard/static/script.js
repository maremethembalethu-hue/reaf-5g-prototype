const colorFor = v => v > 70 ? '#ff6f6f' : v > 45 ? '#ffb44d' : '#33c98f';

async function get(url){
  const r = await fetch(url);
  return r.json();
}

async function loadStats(){
  const s = await get('/api/stats');
  document.getElementById('stat-bars').innerHTML = `
    <div style="flex:1">
      <div class="bar-label">CPU ${s.cpu}%</div>
      <div class="bar-pill"><div class="bar-fill" style="width:${s.cpu}%;background:${colorFor(s.cpu)}"></div></div>
    </div>
    <div style="flex:1">
      <div class="bar-label">RAM ${s.ram}%</div>
      <div class="bar-pill"><div class="bar-fill" style="width:${s.ram}%;background:${colorFor(s.ram)}"></div></div>
    </div>`;
  document.getElementById('stat-list').innerHTML = `
    <div class="item">Model tier<b>${s.tier}</b></div>
    <div class="item">Uptime<b>${s.uptime}</b></div>
    <div class="item">Subnet<b>${s.subnet}</b></div>
    <div class="item">Incidents<b>${s.total_incidents}</b></div>
    <div class="item">Attack events<b>${s.attack_events}</b></div>
    <div class="item">Evidence bundles<b>${s.evidence_bundles}</b></div>
    <div class="item" style="grid-column:span 3">Last detection<b>${s.last_detection}</b></div>`;
}

async function loadLive(){
  const e = await get('/api/live');
  const strip = document.getElementById('live-strip');
  const chip = document.createElement('div');
  chip.className = 'live-chip' + (e.flagged ? ' flag' : '');
  chip.innerHTML = `<b>${e.attack}</b>${e.time} · flow ${e.flow_id ?? '-'}<br>${e.size ?? '-'} pkts · conf ${e.confidence ?? '-'}`;
  strip.prepend(chip);
  while(strip.children.length > 10) strip.removeChild(strip.lastChild);
}

async function loadFeed(){
  const rows = await get('/api/feed');
  document.querySelector('#feed-table tbody').innerHTML = rows.map(r => `
    <tr>
      <td>${r.time}</td><td>${r.flow_id ?? '-'}</td><td>${r.attack}</td>
      <td>${r.confidence}</td><td>${r.window}</td>
      <td>${r.override === 'YES' ? '<span class="badge warn">YES</span>' : 'No'}</td>
      <td>${r.model}</td>
    </tr>`).join('');
}

let timelineChart;
async function loadTimeline(){
  const pts = await get('/api/timeline');
  const ctx = document.getElementById('chart-timeline');
  const data = {
    labels: pts.map(p => p.time),
    datasets: [
      {label:'CPU %', data: pts.map(p=>p.cpu), borderColor:'#7c6cf0', tension:.3, pointRadius:0},
      {label:'RAM %', data: pts.map(p=>p.ram), borderColor:'#33c98f', tension:.3, pointRadius:0},
    ]
  };
  if(timelineChart){ timelineChart.data = data; timelineChart.update(); return; }
  timelineChart = new Chart(ctx, {type:'line', data, options:{responsive:true, plugins:{legend:{position:'bottom'}}}});
}

async function loadCustody(){
  const c = await get('/api/custody');
  document.getElementById('custody-status').innerHTML =
    `${c.intact ? 'R' : 'W'} ${c.total} entries verified · ${c.broken} broken links`;
  document.getElementById('chain-list').innerHTML = c.entries.slice(-12).map(e =>
    `<div class="chain-row"><span>#${e.id} ${e.hash}</span><span>R</span></div>`).join('');
}

async function loadAccuracy(){
  const d = await get('/api/accuracy');
  document.querySelector('#accuracy-table tbody').innerHTML = d.rows.map(r =>
    `<tr><td>${r.level}</td><td>${r.heavy ?? '-'}%</td><td>${r.lite ?? '-'}%</td></tr>`).join('');
  document.getElementById('accuracy-generated').textContent =
    d.generated_at ? `Snapshot from last true_results.py run: ${d.generated_at}` : 'No results generated yet — run true_results.py';
}

async function loadCurrentReplay(){
  const r = await get('/api/current_replay');
  document.getElementById('current-replay').textContent = r.active
    ? `Now replaying: ${r.pcap_path} (expected: ${r.expected_label}) · started ${r.started_at}`
    : 'No replay job currently in progress';
}

async function loadPerAttack(){
  const rows = await get('/api/per_attack');
  document.querySelector('#per-attack-table tbody').innerHTML = rows.map(r =>
    `<tr><td>${r.attack}</td><td>${r.precision}</td><td>${r.recall}</td><td>${r.f1}</td><td>${r.count}</td><td>${r.bundles}</td></tr>`).join('');
}

async function loadOverrides(){
  const d = await get('/api/overrides');
  document.getElementById('override-summary').innerHTML = `
    <div class="chip">Overrides fired<b>${d.total}</b></div>`;
  document.querySelector('#override-table tbody').innerHTML = d.rows.slice(-15).reverse().map(r =>
    `<tr><td>${r.time}</td><td>${r.flow ?? '-'}</td><td>${r.before ?? '(not logged)'}</td><td>${r.after}</td><td>${r.confidence}</td></tr>`).join('');
}

let marginChart;
async function loadMargins(){
  const m = await get('/api/margins');
  const ctx = document.getElementById('chart-margins');
  const data = {labels: m.buckets, datasets:[{label:'Count', data:m.counts, backgroundColor:'#a68cff', borderRadius:6}]};
  if(marginChart){ marginChart.data = data; marginChart.update(); }
  else marginChart = new Chart(ctx, {type:'bar', data, options:{plugins:{legend:{display:false}}}});
  document.getElementById('margin-note').textContent = `Threshold ${m.threshold} · ${m.uncertain} Flood_uncertain windows`;
}

async function loadRuns(){
  const rows = await get('/api/runs');
  document.querySelector('#runs-table tbody').innerHTML = rows.map(r =>
    `<tr><td>${r.id}</td><td>${r.model}</td><td>${r.strict}%</td><td>${r.binary}%</td><td>${r.overrides}</td></tr>`).join('');
}

function refreshAll(){
  loadStats(); loadFeed(); loadTimeline(); loadCustody();
  loadAccuracy(); loadPerAttack(); loadOverrides(); loadMargins(); loadRuns();
  loadCurrentReplay();
}

refreshAll();
setInterval(refreshAll, 10000);
setInterval(loadLive, 2500);
setInterval(loadCurrentReplay, 2500);
