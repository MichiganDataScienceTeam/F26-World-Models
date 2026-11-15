"""
Driving demo with your functions. Scores: metres covered on the road in the last 10 seconds.

python -m labs.driving.demo                 then open http://localhost:8767
python -m labs.driving.demo --solutions     use the solutions

The agent drives; "Drive yourself" hands you the car (arrow keys or WASD). "Show frozen" adds a second agent that
never adapts. Until the six TODOs are filled in, the agent drives without the adapter.
"""

import argparse
import io
import json
import multiprocessing as mp
import os
import queue
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
import torch
from PIL import Image

from envs.driving import car
from envs.driving.road import HALF_WIDTH, Road
from labs.driving import adapter as student
from models.driving import BLOCK, COAST, KEYS, Planner, ahead, load, outputs

PORT = 8767
I3, ZERO = np.eye(3), np.zeros(3)
FAULTS = {"steering reversed": (np.diag([-1., 1., 1.]), ZERO),             # name: (M, b); the car carries out
          "steering pulls left": (I3, np.array([.3, 0., 0.])),            # clip(M c + b), left is positive
          "weak brakes": (np.diag([1., 1., .4]), ZERO)}
LABELS = ("actual fault",) + student.STRUCTURES      # the tiles: the same names as in labs/driving/adapter.py
HAZARD = 1 / 1200    # prior: one fault per 1200 decisions (a minute)
WINDOW = 10.         # seconds of driving in the score
CMDS, LOCK = queue.Queue(), threading.Lock()
LATEST = {"jpg": b"", "gjpg": b"", "n": 0, "state": "{}"}


def planner_process(conn, seed):
    """One car's planner, in its own process so the demo stays in real time. Every decision it gets the car's
    state and runs ONE search round for the next block of keys; after the block's last round it returns the keys."""
    torch.set_num_threads(1)
    model, road = load(), Road(seed)
    new = lambda: Planner(road, model, seed)
    planner = new()
    with torch.no_grad():
        while True:
            try:
                msg = conn.recv()
            except EOFError:                                         # the demo closed
                return
            if msg[0] == "road":
                road = Road(msg[1])
            if msg[0] in ("road", "reset"):
                planner = new()
                continue
            x, cmd, i, M, b = (None if v is None else torch.from_numpy(v) for v in msg[1:6])
            k, j = msg[6:]                                           # keys held now; decision j of the block
            x, cmd, i = ahead(road, model, x, cmd, i, k, M, b, BLOCK - j)       # the car when the next block starts
            planner.round(x[0], cmd[0], i[0], M, b)
            conn.send(planner.commit() if j == BLOCK - 1 else None)


class Car:
    """One car and its planner process."""

    def __init__(self, seed):
        self.conn, child = mp.Pipe()
        self.waiting = False
        mp.Process(target=planner_process, args=(child, seed), daemon=True).start()

    def tell(self, *msg):
        """Tell the planner about a restart or a new road."""
        self.answer()
        self.conn.send(msg)

    def reset(self, road, like=None):
        """Back to the start of the road, or to where another car is."""
        self.tell("reset")
        self.x, self.cmd = (car.start(1, road), torch.zeros(1, 3)) if like is None else (like.x.clone(), like.cmd.clone())
        self.i = torch.zeros(1, dtype=torch.long) if like is None else like.i.clone()
        self.grass = torch.zeros(1, dtype=torch.bool) if like is None else like.grass.clone()
        self.s = 0. if like is None else like.s
        self.keys, self.camera, self.metres = COAST, None if like is None else like.camera, deque(maxlen=int(WINDOW / car.DT))

    def answer(self):
        """Wait for the planner's last round; it returns keys when a block ended."""
        if self.waiting:
            k = self.conn.recv()
            self.waiting = False
            if k is not None:
                self.keys = k

    def plan(self, t, M, b):
        """Start of decision t: take new keys if a block starts, then ask for the next search round."""
        self.answer()
        send = lambda v: None if v is None else np.asarray(v, dtype=np.float32)
        self.conn.send(("tick", self.x.numpy(), self.cmd.numpy(), self.i.numpy(), send(M), send(b), self.keys, t % BLOCK))
        self.waiting = True

    def step(self, road, keys, fault):
        """One decision of the real car. Returns the state and grass flag before it."""
        x0, grass0 = self.x, self.grass
        self.cmd = car.smooth(self.cmd, keys)
        a = self.cmd if fault is None else car.fault(self.cmd, *(torch.tensor(v, dtype=torch.float32)[None] for v in fault))
        self.x = car.drive(self.x, a, self.grass)
        self.i, s, e = road.locate(self.x, self.i)
        self.grass = e.abs() > HALF_WIDTH
        self.metres.append(0. if self.grass.item() else max(s.item() - self.s, 0.))
        self.s = s.item()
        return x0, grass0


class Demo:
    def __init__(self, seed=0, scene=True):
        self.model, self.draw = load(), scene
        self.cars = {name: Car(seed) for name in ("main", "frozen")}
        self.show_frozen, self.human, self.fault, self.error = False, False, None, ""
        self.ms, self.pressed = deque(maxlen=40), torch.zeros(1, 3)
        self.new_road(seed)

    def new_road(self, seed):
        self.seed, self.road = seed, Road(seed)
        for c in self.cars.values():
            c.tell("road", seed)
        if self.draw:
            from envs.driving.scene import Scene
            self.scene = Scene(self.road, seed)
        self.adapter = student.Adapter(self.model, HAZARD)
        self.reset()

    def reset(self):
        """Back to the start; the adapter keeps what it learned."""
        for c in self.cars.values():
            c.reset(self.road)
        self.t = 0

    def sync_frozen(self):
        self.cars["frozen"].reset(self.road, like=self.cars["main"])

    def step(self, keys):
        """One decision. keys (1, 3): your keys, used when you drive."""
        main, frozen = self.cars["main"], self.cars["frozen"]
        fault = FAULTS[self.fault] if self.fault else None
        if not self.human:
            main.plan(self.t, *self.adapter.planner_map())
        if self.show_frozen:
            frozen.plan(self.t, None, None)
        self.pressed = keys if self.human else KEYS[main.keys][None]     # shown on the picture
        x0, grass0 = main.step(self.road, self.pressed, fault)
        if self.show_frozen:
            frozen.step(self.road, KEYS[frozen.keys][None], fault)
        if not self.error:
            try:
                self.adapter.update(x0[0, 3:6], grass0[0], main.cmd[0].double().numpy(), outputs(x0, main.x)[0])
            except NotImplementedError as err:                          # missing TODO: don't adapt
                self.error = str(err)
                self.adapter.theta = self.adapter.theta_nom.copy()
        self.t += 1

    def frame(self, name):
        """One car's picture; each car keeps its own camera."""
        c = self.cars[name]
        self.scene.view_yaw = c.camera
        img = self.scene.render(c.x[0])
        c.camera = self.scene.view_yaw
        return img

    def state(self):
        M, b = FAULTS[self.fault] if self.fault else (I3, ZERO)
        tiles = [{"label": LABELS[0], "M": np.round(M, 2).tolist(), "b": np.round(b, 2).tolist(), "w": None}]
        ad = self.adapter
        maps = ad.structure_maps() if not self.error else [(I3, ZERO)] * 5
        probs = ad.structure_probabilities() if not self.error else np.eye(5)[0]
        for label, (M, b), w in zip(LABELS[1:], maps, probs):
            tiles.append({"label": label, "M": np.round(M, 2).tolist(), "b": np.round(b, 2).tolist(),
                          "w": round(float(w), 3)})
        main = self.cars["main"]
        ms = float(np.median(self.ms)) if self.ms else 0.
        return json.dumps({"fault": self.fault or "", "faults": [""] + list(FAULTS), "frozen": self.show_frozen,
                           "human": self.human, "kmh": round(3.6 * main.x[0, 3].item()),
                           "on": not main.grass.item(), "score": round(sum(main.metres)),
                           "fscore": round(sum(self.cars["frozen"].metres)) if self.show_frozen else 0,
                           "ms": round(ms), "error": self.error, "keys": self.pressed[0].tolist(),
                           "fkeys": KEYS[self.cars["frozen"].keys].tolist(), "tiles": tiles, "commands": ["steer", "throttle", "brake"]})


def jpeg(img):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=85)
    return buf.getvalue()


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Driving adaptation demo</title>
<style>
:root { --bg:#f7f7f5; --ink:#1d1d1f; --muted:#6b6b70; --card:#ffffff; --line:#d9d9d6; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.4 -apple-system, system-ui, sans-serif; }
.wrap { display:flex; gap:20px; padding:18px; flex-wrap:wrap; }
.sim { width:640px; max-width:100%; }
.sim img { width:100%; border-radius:10px; display:block; background:#ddd; }
.view { position:relative; }
#frozenview { margin-top:8px; display:none; }
.pad { position:absolute; left:12px; bottom:12px; display:grid; grid-template-columns:repeat(3, 30px); gap:4px; }
.pad span { height:30px; border-radius:6px; background:rgba(255,255,255,.55); color:#1d1d1f; text-align:center;
            line-height:30px; font-size:15px; transition:background .08s; }
.pad span.on { background:#e8890c; color:#fff; }
.pad .u { grid-column:2; } .pad .l { grid-column:1; }
.panel { flex:1; min-width:420px; max-width:640px; }
.tiles { display:grid; grid-template-columns:repeat(3, 1fr); gap:18px 14px; }
.tile { text-align:center; }
.mat { display:inline-grid; gap:2px; padding:5px; border-radius:6px; background:var(--card);
       outline-style:solid; transition: outline-width .25s, outline-color .25s; }
.cell { width:26px; height:26px; border-radius:2px; }
.sep { width:4px; }
.label { font-size:13px; margin-top:6px; }
.pct { font-size:12px; color:var(--muted); }
.buttons { display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin:12px 0 8px; }
#faultbtns { margin:0 0 18px; }
button { font:inherit; padding:8px 12px; border-radius:8px; border:1px solid var(--line); background:var(--card); cursor:pointer; }
button:hover { border-color:#999; }
button:disabled { color:#aaa; cursor:default; border-color:var(--line); }
button.active { background:#fff1e6; border-color:#c4570c; color:#a3460a; }
button.blue.active { background:#eaf2fb; border-color:#2a6fb5; color:#1f5a96; }
.note { font-size:13px; color:var(--muted); white-space:nowrap; }
.note b { color:var(--ink); font-weight:600; }
.off { color:#b3261e; }
.error { color:#a3460a; font-size:13px; margin-bottom:12px; }
.simbar { display:flex; align-items:center; justify-content:space-between; gap:12px; }
</style></head><body>
<div class="wrap">
  <div class="sim"><div class="view"><img src="/stream" alt="main car"><div class="pad" id="pad"><span class="u">&uarr;</span><span class="l">&larr;</span><span class="d">&darr;</span><span class="r">&rarr;</span></div></div>
    <div class="view" id="frozenview"><img src="/frozen" alt="frozen agent"><div class="pad" id="fpad"><span class="u">&uarr;</span><span class="l">&larr;</span><span class="d">&darr;</span><span class="r">&rarr;</span></div></div>
    <div class="simbar"><div class="buttons">
      <button class="blue" id="humanbtn" onclick="cmd('human')">Drive yourself</button>
      <button class="blue" id="frozenbtn" onclick="cmd('frozen')">Show frozen</button>
      <button onclick="cmd('reset')">Restart</button>
      <button onclick="cmd('road')">New road</button>
    </div><div class="note" id="speed"></div></div>
    <div class="note" id="keys">arrow keys or WASD: steer, throttle, brake</div></div>
  <div class="panel">
    <div class="error" id="error"></div>
    <div class="buttons" id="faultbtns"></div>
    <div class="tiles" id="tiles"></div>
  </div>
</div>
<script>
function lerp(a, b, t) { return a.map((x, i) => Math.round(x + t*(b[i] - x))); }
function col(v) { const t = Math.min(Math.abs(v), 1); const c = lerp([228,228,225], v > 0 ? [232,137,12] : [42,127,212], t);
  return `rgb(${c[0]},${c[1]},${c[2]})`; }
function cmd(c) { fetch('/cmd?c=' + encodeURIComponent(c), {method: 'POST'}); }
const held = new Set();
const KEYMAP = {ArrowLeft:'l', a:'l', ArrowRight:'r', d:'r', ArrowUp:'u', w:'u', ArrowDown:'d', s:'d'};
function send() { const steer = (held.has('l') ? 1 : 0) - (held.has('r') ? 1 : 0);
  cmd('keys:' + steer + ',' + (held.has('u') ? 1 : 0) + ',' + (held.has('d') ? 1 : 0)); }
function pad(id, k) {   // k = [steer, throttle, brake]; steer +1 is left
  const on = {l: k[0] > 0, r: k[0] < 0, u: k[1] > 0, d: k[2] > 0};
  for (const c of 'lrud') document.querySelector('#' + id + ' .' + c).classList.toggle('on', on[c]); }
addEventListener('keydown', e => { const k = KEYMAP[e.key]; if (!k) return; e.preventDefault();
  if (!held.has(k)) { held.add(k); send(); } });
addEventListener('keyup', e => { const k = KEYMAP[e.key]; if (!k) return; held.delete(k); send(); });
addEventListener('blur', () => { held.clear(); send(); });
let built = false;
function render(s) {
  document.getElementById('error').textContent = s.error ? s.error + ' (running without the adapter)' : '';
  document.getElementById('speed').innerHTML = `<span class="${s.on ? '' : 'off'}">${s.kmh} km/h</span> &nbsp;` +
    `${s.human ? 'you' : 'agent'} <b>${s.score} m</b>` + (s.frozen ? ` &nbsp;frozen <b>${s.fscore} m</b>` : '') + ` &nbsp;&nbsp;${s.ms} ms`;
  const fb = document.getElementById('faultbtns');
  if (!built) { fb.innerHTML = s.faults.map((f, i) => `<button id="f${i}" onclick="cmd('fault${i}')">${f || 'None'}</button>`).join(''); built = true; }
  s.faults.forEach((f, i) => document.getElementById('f' + i).classList.toggle('active', s.fault === f));
  const hb = document.getElementById('humanbtn');
  hb.classList.toggle('active', s.human); hb.textContent = s.human ? 'Let the agent drive' : 'Drive yourself';
  const fbtn = document.getElementById('frozenbtn');
  fbtn.classList.toggle('active', s.frozen); fbtn.textContent = s.frozen ? 'Hide frozen' : 'Show frozen';
  document.getElementById('frozenview').style.display = s.frozen ? 'block' : 'none';
  pad('pad', s.keys); pad('fpad', s.fkeys);
  document.getElementById('keys').style.visibility = s.human ? 'visible' : 'hidden';
  document.getElementById('tiles').innerHTML = s.tiles.map(t => {   // the map: matrix M | offsets b
    const A = t.b.length; let cells = '';
    for (let i = 0; i < A; i++) { for (let j = 0; j < A; j++) cells += `<div class="cell" title="${s.commands[j]} command -> ${s.commands[i]}: ${t.M[i][j]}" style="background:${col(t.M[i][j])}"></div>`;
      cells += `<div class="sep"></div><div class="cell" title="${s.commands[i]} offset: ${t.b[i]}" style="background:${col(t.b[i])}"></div>`; }
    const w = t.w, ow = w === null ? 1 : 1 + Math.round(5*w);
    const oc = w === null ? '#9a9a96' : `rgb(${lerp([205,205,200],[46,158,74],w).join(',')})`;
    return `<div class="tile"><div class="mat" style="grid-template-columns:repeat(${A}, 26px) 4px 26px; outline-width:${ow}px; outline-color:${oc}">${cells}</div>
      <div class="label">${t.label}</div><div class="pct">${w === null ? '&nbsp;' : Math.round(100*w) + '%'}</div></div>`;
  }).join('');
}
setInterval(() => fetch('/state').then(r => r.json()).then(render).catch(() => {}), 100);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, body, kind):
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self.send(PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/state":
            with LOCK:
                self.send(LATEST["state"].encode(), "application/json")
        elif self.path in ("/stream", "/frozen"):           # MJPEG streams
            key = "jpg" if self.path == "/stream" else "gjpg"
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last = -1
            try:
                while True:
                    with LOCK:
                        n, jpg = LATEST["n"], LATEST[key]
                    if n != last and jpg:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                        last = n
                    time.sleep(.005)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path.startswith("/cmd?c="):
            CMDS.put(self.path.split("=", 1)[1].replace("%3A", ":").replace("%2C", ","))
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solutions", action="store_true", help="use solutions/driving_adapter.py")
    solutions = parser.parse_args().solutions
    if solutions:
        student.use_solutions()
    torch.set_num_threads(1)
    demo = Demo()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"demo running: open http://localhost:{PORT}", flush=True)
    keys, n = torch.zeros(1, 3), 0
    while True:                                          # simulation and rendering stay on the main thread
        tick = time.perf_counter()
        now = keys.clone()
        while not CMDS.empty():
            c = CMDS.get()
            if c.startswith("keys:"):
                keys = torch.tensor([[float(v) for v in c[5:].split(",")]])
                now = torch.where(keys.abs() > now.abs(), keys, now)     # a short tap still counts once
            elif c.startswith("fault") and c[5:].isdigit() and int(c[5:]) <= len(FAULTS):
                demo.fault = ([None] + list(FAULTS))[int(c[5:])]
            elif c == "frozen":
                demo.show_frozen = not demo.show_frozen
                demo.sync_frozen()
            elif c == "human":
                demo.human = not demo.human
                demo.cars["main"].tell("reset")
            elif c == "reset":
                demo.reset()
            elif c == "road":
                demo.new_road(demo.seed + 1)
        demo.step(now)
        jpg = jpeg(demo.frame("main"))
        gjpg = jpeg(demo.frame("frozen")) if demo.show_frozen else b""
        n += 1
        demo.ms.append(1000 * (time.perf_counter() - tick))      # decision + rendering
        with LOCK:
            LATEST["jpg"], LATEST["gjpg"], LATEST["n"], LATEST["state"] = jpg, gjpg, n, demo.state()
        time.sleep(max(0., car.DT - (time.perf_counter() - tick)))


if __name__ == "__main__":
    main()
