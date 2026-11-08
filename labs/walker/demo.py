"""
Walker demo with your adapter. Scores: reward over the last second (max 40).

python -m labs.walker.demo                 then open http://localhost:8765
python -m labs.walker.demo --solutions     use the solutions
"""

import argparse
import io
import json
import queue
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image

from envs.walker import MOTORS, WalkerEnv
from labs.walker import adapter as student
from models.tdmpc2 import Planner, load

PORT = 8765
HORIZON = 6          # planning steps (measured: 3 is too short with faults)
FAULTS = {1: ("left knee jammed bent", {4: (0., .8)}), 2: ("left knee dead (floppy)", {4: (0., 0.)}),
          3: ("right knee dead (floppy)", {1: (0., 0.)})}
LABELS = ("actual fault", "no change", "one gain for all", "gain per motor", "gain + offset", "any mixing")
HAZARD = 1 / 500     # prior: one fault per 500 decisions
CMDS, LOCK = queue.Queue(), threading.Lock()
LATEST = {"jpg": b"", "gjpg": b"", "n": 0, "state": "{}"}


class Demo:
    def __init__(self):
        dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model, self.cpu_model = load(device=dev), load()          # planner on GPU, adapter on CPU
        self.env, self.frozen_env = WalkerEnv(seed=0), WalkerEnv(seed=1)
        self.planners = {B: Planner(self.model, B, HORIZON) for B in (1, 2)}
        self.dt = .05                                                   # seconds per decision
        self.show_frozen, self.fault, self.error = False, 0, ""
        self.adapter = student.Adapter(self.cpu_model, HAZARD)
        self.ms, self.last_B = deque(maxlen=40), 1
        self.reset()

    def reset(self):
        """New episode; the adapter keeps what it learned."""
        self.obs, self.first, self.rewards, self.z = self.env.reset(), True, deque(maxlen=20), None
        self.sync_frozen()

    def sync_frozen(self):
        self.frozen_env.copy_state_from(self.env)
        self.fobs, self.ffirst, self.frewards = self.obs.copy(), True, deque(maxlen=20)

    def set_fault(self, k):
        gain, offset = np.ones(6), np.zeros(6)
        for i, (g, o) in (FAULTS[k][1] if k else {}).items():
            gain[i], offset[i] = g, o
        for e in (self.env, self.frozen_env):
            e.set_fault(gain, offset)
        self.fault = k
        self.sync_frozen()

    def step(self):
        B = 2 if self.show_frozen else 1
        pl = self.planners[B]
        if B != self.last_B:                                            # keep the warm start
            pl.prev_mean[0] = self.planners[self.last_B].prev_mean[0]
            self.last_B = B
        M, b = self.adapter.map()
        pl.M[0], pl.b[0] = torch.tensor(M, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)  # row 1: frozen
        obs = np.stack([self.obs, self.fobs][:B])
        c = pl.act(obs, [self.first, self.ffirst][:B])
        nobs, r = self.env.step(c[0])
        if not self.error:
            with torch.no_grad():
                z = self.cpu_model.encode(torch.tensor(self.obs)).numpy() if self.z is None else self.z
                nz = self.cpu_model.encode(torch.tensor(nobs)).numpy()
            try:
                self.adapter.update(z, c[0].astype(np.float64), nz)
            except NotImplementedError as err:                          # missing TODO: don't adapt
                self.error = str(err)
                self.adapter.theta = self.adapter.theta_nom.copy()
            self.z = nz
        self.obs, self.first = nobs, False
        self.rewards.append(r)
        if self.show_frozen:
            self.fobs, fr = self.frozen_env.step(c[1])
            self.ffirst = False
            self.frewards.append(fr)

    def state(self):
        tiles = [{"label": LABELS[0], "M": np.diag(self.env.gain).round(2).tolist(),
                  "b": self.env.offset.round(2).tolist(), "w": None}]
        ad = self.adapter
        maps = ad.structure_maps() if not self.error else [ad.map()] * 5
        probs = ad.structure_probabilities() if not self.error else np.eye(5)[0]
        for label, (M, b), w in zip(LABELS[1:], maps, probs):
            tiles.append({"label": label, "M": np.round(M, 2).tolist(), "b": np.round(b, 2).tolist(),
                          "w": round(float(w), 3)})
        ms = float(np.median(self.ms)) if self.ms else 0.
        return json.dumps({"fault": self.fault, "faults": [[0, "None"]] + [[k, v[0]] for k, v in FAULTS.items()],
                           "frozen": self.show_frozen, "score": int(round(sum(self.rewards))),
                           "fscore": int(round(sum(self.frewards))), "fps": int(round(1000 / ms)) if ms else 0,
                           "error": self.error, "tiles": tiles, "motors": MOTORS})


def jpeg(img):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=85)
    return buf.getvalue()


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Walker adaptation demo</title>
<style>
:root { --bg:#f7f7f5; --ink:#1d1d1f; --muted:#6b6b70; --card:#ffffff; --line:#d9d9d6; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.4 -apple-system, system-ui, sans-serif; }
.wrap { display:flex; gap:20px; padding:18px; flex-wrap:wrap; }
.sim { width:640px; max-width:100%; }
.sim img { width:100%; border-radius:10px; display:block; background:#ddd; }
#frozen { margin-top:8px; display:none; }
.panel { flex:1; min-width:420px; max-width:640px; }
.tiles { display:grid; grid-template-columns:repeat(3, 1fr); gap:18px 14px; }
.tile { text-align:center; }
.mat { display:inline-grid; gap:2px; padding:5px; border-radius:6px; background:var(--card);
       outline-style:solid; transition: outline-width .25s, outline-color .25s; }
.cell { width:17px; height:17px; border-radius:2px; }
.sep { width:4px; }
.label { font-size:13px; margin-top:6px; }
.pct { font-size:12px; color:var(--muted); }
.buttons { display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin:12px 0 8px; }
#faultbtns { margin:0 0 18px; }
button { font:inherit; padding:8px 12px; border-radius:8px; border:1px solid var(--line); background:var(--card); cursor:pointer; }
button:hover { border-color:#999; }
button.active { background:#fff1e6; border-color:#c4570c; color:#a3460a; }
button.frozen.active { background:#eaf2fb; border-color:#2a6fb5; color:#1f5a96; }
.note { font-size:13px; color:var(--muted); white-space:nowrap; }
.note b { color:var(--ink); font-weight:600; }
.error { color:#a3460a; font-size:13px; margin-bottom:12px; }
.simbar { display:flex; align-items:center; justify-content:space-between; gap:12px; }
</style></head><body>
<div class="wrap">
  <div class="sim"><img src="/stream" alt="adaptive agent"><img id="frozen" src="/frozen" alt="frozen agent">
    <div class="simbar"><div class="buttons">
      <button class="frozen" id="frozenbtn" onclick="cmd('frozen')">Show frozen</button>
      <button onclick="cmd('reset')">Reset episode</button>
    </div><div class="note" id="speed"></div></div></div>
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
function cmd(c) { fetch('/cmd?c=' + c, {method: 'POST'}); }
let built = false;
function render(s) {
  document.getElementById('error').textContent = s.error ? s.error + ' (running without the adapter)' : '';
  document.getElementById('speed').innerHTML = `adapter <b>${s.score}</b>` + (s.frozen ? ` &nbsp;frozen <b>${s.fscore}</b>` : '')
    + ` &nbsp;&nbsp;${s.fps} fps`;
  const fb = document.getElementById('faultbtns');
  if (!built) { fb.innerHTML = s.faults.map(f => `<button id="f${f[0]}" onclick="cmd('fault${f[0]}')">${f[1]}</button>`).join(''); built = true; }
  s.faults.forEach(f => document.getElementById('f' + f[0]).classList.toggle('active', s.fault === f[0]));
  const fbtn = document.getElementById('frozenbtn');
  fbtn.classList.toggle('active', s.frozen); fbtn.textContent = s.frozen ? 'Hide frozen' : 'Show frozen';
  document.getElementById('frozen').style.display = s.frozen ? 'block' : 'none';
  document.getElementById('tiles').innerHTML = s.tiles.map(t => {   // the map: matrix M | offsets b
    const A = t.b.length; let cells = '';
    for (let i = 0; i < A; i++) { for (let j = 0; j < A; j++) cells += `<div class="cell" style="background:${col(t.M[i][j])}"></div>`;
      cells += `<div class="sep"></div><div class="cell" style="background:${col(t.b[i])}"></div>`; }
    const w = t.w, ow = w === null ? 1 : 1 + Math.round(5*w);
    const oc = w === null ? '#9a9a96' : `rgb(${lerp([205,205,200],[46,158,74],w).join(',')})`;
    return `<div class="tile"><div class="mat" style="grid-template-columns:repeat(${A}, 17px) 4px 17px; outline-width:${ow}px; outline-color:${oc}">${cells}</div>
      <div class="label">${t.label}</div><div class="pct">${w === null ? '&nbsp;' : Math.round(100*w) + '%'}</div></div>`;
  }).join('');
}
setInterval(() => fetch('/state').then(r => r.json()).then(render).catch(() => {}), 200);
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
                    time.sleep(.01)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path.startswith("/cmd?c="):
            CMDS.put(self.path.split("=", 1)[1])
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solutions", action="store_true", help="use solutions/walker_adapter.py")
    if parser.parse_args().solutions:
        student.use_solutions()
    demo = Demo()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"demo running: open http://localhost:{PORT}", flush=True)
    n = 0
    while True:                                          # simulation and rendering stay on the main thread
        tick = time.perf_counter()
        while not CMDS.empty():
            c = CMDS.get()
            if c.startswith("fault") and c[5:].isdigit() and (int(c[5:]) in FAULTS or c[5:] == "0"):
                demo.set_fault(int(c[5:]))
            elif c == "frozen":
                demo.show_frozen = not demo.show_frozen
                demo.sync_frozen()
            elif c == "reset":
                demo.reset()
        demo.step()
        jpg = jpeg(demo.env.render())
        gjpg = jpeg(demo.frozen_env.render()) if demo.show_frozen else b""
        n += 1
        demo.ms.append(1000 * (time.perf_counter() - tick))      # decision + rendering
        with LOCK:
            LATEST["jpg"], LATEST["gjpg"], LATEST["n"], LATEST["state"] = jpg, gjpg, n, demo.state()
        time.sleep(max(0., demo.dt - (time.perf_counter() - tick)))


if __name__ == "__main__":
    main()
