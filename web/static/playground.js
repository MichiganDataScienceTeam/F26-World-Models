"use strict";

const $ = id => document.getElementById(id);
const ink = "#222222", accent = "#1155cc";
const modes = ["frozen", "bayes", "oracle"];

let state = null;
let running = false, runToken = 0;
let queue = Promise.resolve();

function pause() {
  running = false;
  runToken++;
  $("play").textContent = "Play";
}

function failure(error) {
  pause();
  $("message").textContent = error.message;
  $("message").hidden = false;
}

function api(path, payload, synchronize = false) {
  queue = queue.catch(() => {}).then(async () => {
    const options = payload === undefined ? {} : {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    };
    const response = await fetch(path, options);
    const data = await response.json();

    if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
    update(data, synchronize);
    return data;
  });
  return queue;
}

function buildControls() {
  $("physics").replaceChildren();
  for (const [key, text, min, max, step, unit] of state.physics_controls) {
    const label = document.createElement("label"), output = document.createElement("output");
    const input = document.createElement("input");

    label.className = "slider-label";
    label.append(document.createTextNode(text));
    input.type = "range";
    input.min = min; input.max = max; input.step = step;
    input.value = state.physics[key]; input.dataset.key = key;

    const show = () => output.textContent = `${Number(input.value).toFixed(2)} ${unit}`;
    input.addEventListener("input", show);
    input.addEventListener("change", () => {
      api("/api/configure", {physics: {[key]: Number(input.value)}}).catch(failure);
    });

    label.append(output, input);
    $("physics").append(label);
    show();
  }
}

function update(data, synchronize) {
  const changedEnvironment = !state || data.environment !== state.environment;
  state = data;

  if (changedEnvironment) buildControls();
  if (changedEnvironment || synchronize) {
    $("environment").value = state.environment;
    for (const input of $("physics").querySelectorAll("input")) {
      input.value = state.physics[input.dataset.key];
      input.dispatchEvent(new Event("input"));
    }
  }

  $("clock").textContent = `${state.time.toFixed(2)} s`;
  $("reset-on-change").setAttribute("aria-pressed", String(state.reset_on_change));
  for (const mode of modes) $(`${mode}-status`).textContent = state.runs[mode].status;
  for (const id of ["play", "reset", "restore", "reset-posterior", "reset-on-change"]) $(id).disabled = false;
  $("message").hidden = true;
  draw();
}

async function loop(token) {
  if (!running || token !== runToken) return;
  const start = performance.now();

  try { await api("/api/step", {steps: 2}); }
  catch (error) { failure(error); return; }

  if (running && token === runToken) {
    const delay = 2 * state.dt * 1000 - (performance.now() - start);
    setTimeout(() => loop(token), Math.max(0, delay));
  }
}

function canvas(id) {
  const node = $(id), rect = node.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.round(rect.width * ratio), height = Math.round(rect.height * ratio);

  if (node.width !== width || node.height !== height) {
    node.width = width; node.height = height;
  }

  const ctx = node.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.font = "11px monospace"; ctx.lineCap = "round"; ctx.lineJoin = "round";
  return {ctx, width: rect.width, height: rect.height};
}

function line(ctx, points, color = ink, width = 1, dash = []) {
  if (points.length < 2) return;
  ctx.strokeStyle = color; ctx.lineWidth = width;
  ctx.setLineDash(dash); ctx.beginPath();

  points.forEach(([x, y], index) => {
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });

  ctx.stroke(); ctx.setLineDash([]);
}

function goal(ctx, x, y) {
  line(ctx, [[x - 7, y], [x + 7, y]], accent);
  line(ctx, [[x, y - 7], [x, y + 7]], accent);
}

function drawDrone(mode) {
  const {ctx, width, height} = canvas(mode), run = state.runs[mode];
  if (!run.active) return;

  const scale = Math.min((width - 20) / 6, (height - 20) / 4);
  const point = (x, z) => [width / 2 + x * scale, height - 10 - z * scale];
  line(ctx, [point(-3, 0), point(3, 0)]);
  ctx.globalAlpha = .25;
  line(ctx, run.trace.map(row => point(...row)));
  ctx.globalAlpha = 1;
  line(ctx, run.predicted_path.map(row => point(row[0], row[1])), ink, 1, [3, 4]);
  goal(ctx, ...point(...run.target));

  const [x, z, theta] = run.physical_state;
  const [px, py] = point(x, z), arm = Math.max(9, .2 * scale);
  ctx.save(); ctx.translate(px, py); ctx.rotate(-theta);
  line(ctx, [[-arm, 0], [arm, 0]], ink, 2);
  ctx.fillStyle = ink; ctx.fillRect(-4, -3, 8, 6);
  for (const rx of [-arm, arm]) line(ctx, [[rx - 6, -4], [rx + 6, -4]], ink, 2);
  ctx.restore();
}

function drawPendulum(mode) {
  const {ctx, width, height} = canvas(mode), run = state.runs[mode];
  if (!run.active) return;

  const cx = width / 2, cy = height / 2;
  const radius = Math.min((height - 24) / 2.2, (width - 24) / 2.2) * state.physics.length;
  const point = theta => [cx + radius * Math.sin(theta), cy - radius * Math.cos(theta)];
  line(ctx, run.predicted_path.map(row => point(Math.atan2(row[1], row[0]))), ink, 1, [3, 4]);
  goal(ctx, ...point(0));

  const tip = point(run.physical_state[0]);
  line(ctx, [[cx, cy], tip], ink, 2);
  ctx.fillStyle = ink;
  ctx.beginPath(); ctx.arc(...tip, 6, 0, 2 * Math.PI); ctx.fill();
  ctx.beginPath(); ctx.arc(cx, cy, 3, 0, 2 * Math.PI); ctx.fill();
}

function drawChart() {
  const {ctx, width, height} = canvas("chart"), rows = state.history;
  const left = 68, right = width - 8, top = 10, bottom = height - 24;
  line(ctx, [[left, top], [left, bottom], [right, bottom]]);
  if (rows.length < 2) return;

  const values = rows.flatMap(row => modes.map(mode => row[mode])).filter(Number.isFinite);
  const high = Math.max(1e-10, ...values) * 1.08;
  const t0 = rows[0].time, t1 = rows.at(-1).time;
  const point = (time, value) => [
    left + (time - t0) / (t1 - t0) * (right - left),
    bottom - value / high * (bottom - top)
  ];

  ctx.fillStyle = ink;
  for (let i = 0; i <= 3; i++) {
    const value = high * i / 3, y = point(t0, value)[1];
    line(ctx, [[left - 3, y], [left, y]]);
    ctx.fillText(value === 0 ? "0" : value.toExponential(1), 2, y + 4);
  }
  ctx.fillText(`${t0.toFixed(1)} s`, left, height - 5);
  ctx.fillText(`${t1.toFixed(1)} s`, right - 45, height - 5);

  for (const mode of modes) {
    const color = mode === "bayes" ? accent : ink, width = mode === "bayes" ? 1.7 : 1.2;
    const dash = mode === "oracle" ? [4, 4] : [];
    let segment = [];

    for (const row of rows) {
      if (Number.isFinite(row[mode])) {
        segment.push(point(row.time, row[mode]));
      } else {
        line(ctx, segment, color, width, dash);
        segment = [];
      }
    }
    line(ctx, segment, color, width, dash);
  }
}

function draw() {
  if (!state) return;
  const drawEnvironment = state.environment === "drone" ? drawDrone : drawPendulum;
  modes.forEach(drawEnvironment);
  drawChart();
}

$("play").addEventListener("click", () => {
  if (running) return pause();
  running = true;
  $("play").textContent = "Pause";
  loop(++runToken);
});
$("reset").addEventListener("click", () => {
  pause();
  api("/api/reset", {}, true).catch(failure);
});
$("environment").addEventListener("change", () => {
  pause();
  api("/api/reset", {environment: $("environment").value}, true).catch(failure);
});
$("restore").addEventListener("click", () => api("/api/restore", {}, true).catch(failure));
$("reset-posterior").addEventListener("click", () => api("/api/reset-posterior", {}).catch(failure));
$("reset-on-change").addEventListener("click", () => {
  const enabled = $("reset-on-change").getAttribute("aria-pressed") !== "true";
  $("reset-on-change").setAttribute("aria-pressed", String(enabled));
  api("/api/configure", {reset_on_change: enabled}).catch(failure);
});

window.addEventListener("resize", draw);
api("/api/state", undefined, true).catch(failure);
