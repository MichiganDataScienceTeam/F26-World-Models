"""Batched arcade car (torch, any batch size), like a driving game: it turns where the keys say, with a grip limit.

Speed, engine and brakes use a real car's numbers (BMW 320i, CommonRoad vehicle 2; Althoff & Wuersching 2020).
Turning is arcade: the car's heading turns at the rate the keys ask for, a bit more under braking (weight moves
onto the front tyres) and a bit less under throttle. The tyres then push the car sideways to follow its heading, but
only up to the grip that braking or throttle leaves over (friction circle). Asking for more than that makes the car
slide: drifts on full steer at speed; braking into a corner turns the nose in and lets the car slide.
Brakes have no ABS: the brakes can ask for more than the grip (A_MAX > GRIP), and then the wheels lock (or spin,
on throttle): they slow the car only by sliding friction, and the grip left for turning is what that leaves.

State (B, 7): X, Y, yaw, vx (forward speed), vy (sideways speed, left positive), r (yaw rate), delta (front-wheel
angle, for drawing only). Commands (B, 3) after smoothing and the fault: steer in [-1, 1], throttle and brake in [0, 1].
"""
import math

import torch

G = 9.81
# CommonRoad vehicle 2 (published)
M, LF, LR, H_CG = 1093.2952334674046, 1.1561957064, 1.4227170936, 0.5748689544   # H_CG: centre of gravity height
A_MAX, V_SWITCH, V_MAX = 11.5, 7.319, 50.8      # accel/brake limit [m/s^2]; above V_SWITCH accel ~ 1/v (power)
MU = 1.0489                                     # tyre friction
GRIP = MU*G                                     # largest tyre acceleration, any direction [m/s^2]
L = LF + LR
# Typical sedan turning circle; sliding friction about 80% of peak (rubber on asphalt); drag and rolling
# resistance of a mid-size sedan. Grass: half engine power and heavy rolling resistance (about 50 km/h at most).
R_MIN = 5.5                                     # tightest turn radius [m]
SLIDE = .8                                      # sliding friction / peak friction
CDA, RHO = 0.29*2.1, 1.2                        # drag coefficient x frontal area [m^2], air density [kg/m^3]
ROLL_ROAD, ROLL_GRASS, ENGINE_GRASS = .015, .3, .5
# Full steer at speed asks for this multiple of the grip limit, so a held full steer drifts a little.
OVER = 1.3


def step(x, cmd, offroad, dt):
    """One explicit-Euler step of length dt. offroad: (B,) bool."""
    X, Y, yaw, vx, vy, r, _ = x.unbind(1)
    steer, thr, brk = cmd.unbind(1)
    roll = torch.where(offroad, torch.full_like(vx, ROLL_GRASS), torch.full_like(vx, ROLL_ROAD))
    engine = torch.where(offroad, torch.full_like(vx, ENGINE_GRASS), torch.ones_like(vx))
    # what the tyres are asked for along the car: engine (power-limited above V_SWITCH) minus brakes
    drive = thr*engine*A_MAX*torch.clamp(V_SWITCH/vx.clamp(min=1e-3), max=1.)*(vx < V_MAX)
    ask = drive - brk*A_MAX*torch.tanh(vx/.5)
    spin = ask.abs() > GRIP                                         # wheels lock (braking) or spin (throttle)
    ax = torch.where(spin, SLIDE*GRIP*torch.sign(ask), ask)
    side = torch.sqrt((GRIP**2 - ax**2).clamp(min=0.))              # grip left for cornering
    # heading: turns as the keys ask (tightest turn at low speed, OVER x grip at high speed). Weight shift: braking
    # loads the front tyres (by ax*H_CG/(G*L) of the weight), so the nose turns in more; throttle unloads them.
    want = steer*torch.minimum(vx/R_MIN, OVER*GRIP/vx.clamp(min=1e-3))
    r = want*(1 - ax*H_CG/(G*L))
    delta = torch.atan(L*want/vx.clamp(min=.5))
    # sideways: tyres push the car's velocity towards its heading, within the grip left
    ay = (vx*r - vy/dt).clamp(-side, side)
    resist = (.5*RHO*CDA*vx*vx.abs()/M + roll*G)*torch.tanh(vx/.5)
    vx_n = (vx + dt*(ax - resist + vy*r)).clamp(min=0.)             # no reversing
    vy_n = vy + dt*(ay - vx*r)
    X = X + dt*(vx*torch.cos(yaw) - vy*torch.sin(yaw))
    Y = Y + dt*(vx*torch.sin(yaw) + vy*torch.cos(yaw))
    yaw = yaw + dt*r
    return torch.stack([X, Y, yaw, vx_n, vy_n, r, delta], 1)


# Keys and smoothing (the human's and the agent's only interface).
# Keys (B, 3): steer key in {-1, 0, 1}, throttle key and brake key in {0, 1}. Each smoothed command ramps towards
# its key at a fixed rate. Full travel takes RAMP seconds, like a game's key smoothing.
RAMP = .2
DT, SUB = .05, 5                                                      # decision length [s], physics substeps


def smooth(cmd, keys, dt=DT):
    return cmd + (keys - cmd).clamp(-dt/RAMP, dt/RAMP)


def fault(cmd, M, b):
    """Executed commands a = clip(M c + b), each command clipped to its own range. M (B, 3, 3), b (B, 3)."""
    a = (M@cmd.unsqueeze(-1)).squeeze(-1) + b
    return torch.stack([a[:, 0].clamp(-1, 1), a[:, 1].clamp(0, 1), a[:, 2].clamp(0, 1)], 1)


def drive(x, a, grass):
    """One decision with executed commands a (B, 3): SUB physics substeps. grass: (B,) bool."""
    for _ in range(SUB):
        x = step(x, a, grass, DT/SUB)
    return x


def decide(x, cmd, keys, grass, M=None, b=None):
    """One decision: smooth the keys, apply the command map, drive. Returns x, cmd."""
    cmd = smooth(cmd, keys)
    return drive(x, cmd if M is None else fault(cmd, M, b), grass), cmd


def start(B, road, v0=0.):
    """B cars at the road's start, facing along it."""
    x = torch.zeros(B, 7)
    x[:, 0], x[:, 1], x[:, 2], x[:, 3] = road.x[0], road.y[0], road.yaw[0], v0
    return x
