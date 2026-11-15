"""3D view of the driving world: MuJoCo's off-screen renderer, low-poly scenery, chase camera.

MuJoCo here only draws; the car's physics is car.py (flat, 2D). Low-poly look = flat shading: every triangle has its
own vertices and normal. Render cost grows with the number of geoms, so everything static of one colour is merged
into a few big meshes.

Terrain: hills from smooth random noise, flat (slightly below the road) within FLAT metres of the road, rising with
distance into mountains at the edge of the world. Trees grow in clusters: pines, round trees (some autumn), birches,
bushes and rocks.
"""
import numpy as np
import mujoco
from scipy.ndimage import zoom
from scipy.spatial import cKDTree

from envs.driving.road import HALF_WIDTH

W, H = 960, 540
CELL, FLAT, MARGIN = 20., 25., 1300.        # terrain grid [m]; flat corridor half-width [m]; world beyond the road [m]


def flat_mesh(v, f, out=None):
    """Per-face vertices and normals (flat shading). Faces are turned to face up (out=None) or away from out."""
    v, f = np.asarray(v, float), np.asarray(f)
    tri = v[f]                                                       # (F, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    away = tri.mean(1) - (out if out is not None else tri.mean(1) - [0, 0, 1])
    flip = (n*away).sum(1) < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    n[flip] *= -1
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return tri.reshape(-1, 3), np.repeat(n, 3, 0)


class Merger:
    """Collects flat-shaded triangles per colour, then emits them as a few meshes."""
    def __init__(self):
        self.parts = {}

    def add(self, col, v, f, out=None):
        V, N = flat_mesh(v, f, out)
        self.parts.setdefault(col, []).append((V, N))

    def xml(self, prefix='m', geom_extra=''):
        assets, geoms = '', ''
        for j, (col, parts) in enumerate(self.parts.items()):
            V = np.concatenate([p[0] for p in parts])
            N = np.concatenate([p[1] for p in parts])
            for c0 in range(0, len(V), 60000):                      # keep each mesh a manageable size
                v, n = V[c0:c0 + 60000], N[c0:c0 + 60000]
                name = f'{prefix}{j}_{c0}'
                assets += (f'<mesh name="{name}" inertia="shell" vertex="{" ".join(f"{a:.3f}" for a in v.ravel())}" '
                           f'normal="{" ".join(f"{a:.2f}" for a in n.ravel())}" '
                           f'face="{" ".join(map(str, range(len(v))))}"/>')
                geoms += f'<geom type="mesh" mesh="{name}" {col} contype="0" conaffinity="0"{geom_extra}/>'
        return assets, geoms


def rgba(r, g, b):
    return f'rgba="{r:.3f} {g:.3f} {b:.3f} 1"'


# ---- shapes (vertices, faces) ----
def strip(x, y, yaw, e0, e1, z):
    """Flat ribbon between lateral offsets e0 < e1 along a centre line."""
    nx, ny = -np.sin(yaw), np.cos(yaw)
    R = np.c_[x + e0*nx, y + e0*ny, np.full(len(x), z)]
    L = np.c_[x + e1*nx, y + e1*ny, np.full(len(x), z)]
    n, i = len(x), np.arange(len(x) - 1)
    return np.r_[R, L], np.r_[np.c_[i, i + 1, n + i + 1], np.c_[i, n + i + 1, n + i]]


def cone(k, r, h, z0=0.):
    a = np.arange(k)*2*np.pi/k
    v = np.r_[np.c_[r*np.cos(a), r*np.sin(a), np.full(k, z0)], [[0, 0, z0 + h]]]
    return v, np.array([[i, (i + 1) % k, k] for i in range(k)] + [[0, i + 1, i] for i in range(1, k - 1)])


def prism(k, r, z0, z1, r1=None):
    """k-sided column from z0 to z1 (radius r at the bottom, r1 at the top)."""
    a = np.arange(k)*2*np.pi/k
    r1 = r if r1 is None else r1
    v = np.r_[np.c_[r*np.cos(a), r*np.sin(a), np.full(k, z0)], np.c_[r1*np.cos(a), r1*np.sin(a), np.full(k, z1)]]
    f = [[i, (i + 1) % k, k + (i + 1) % k] for i in range(k)] + [[i, k + (i + 1) % k, k + i] for i in range(k)]
    f += [[0, i + 1, i] for i in range(1, k - 1)] + [[k, k + i, k + i + 1] for i in range(1, k - 1)]
    return v, np.array(f)


_P = (1 + 5**.5)/2
ICO_V = np.array([[-1, _P, 0], [1, _P, 0], [-1, -_P, 0], [1, -_P, 0], [0, -1, _P], [0, 1, _P], [0, -1, -_P],
                  [0, 1, -_P], [_P, 0, -1], [_P, 0, 1], [-_P, 0, -1], [-_P, 0, 1]])/np.sqrt(1 + _P**2)
ICO_F = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
                  [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5],
                  [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])


def blob(rng, rx, ry, rz, z0, jitter=.15):
    """Lumpy icosahedron: tree crowns, bushes, rocks."""
    v = ICO_V*(1 + rng.uniform(-jitter, jitter, (12, 1)))*[rx, ry, rz] + [0, 0, z0]
    return v, ICO_F


def placed(vf, pos, yaw):
    v, f = vf
    c, s = np.cos(yaw), np.sin(yaw)
    return v@np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) + pos, f


def loft(sections):
    """Body from cross-sections along x. Each section: (x, half-width, bottom z, top z, bevel). Returns v, f, and
    for each face which side it is on ('top', 'side', 'end')."""
    ring = lambda x, w, z0, z1, b: [[x, -w, z0], [x, w, z0], [x, w, z1 - b], [x, w - b, z1], [x, -(w - b), z1],
                                    [x, -w, z1 - b]]
    v = np.array([p for s in sections for p in ring(*s)], float)
    f, kind = [], []
    names = ['bottom', 'side', 'top', 'top', 'top', 'side']             # edge i -> i+1 of the ring
    for j in range(len(sections) - 1):
        a, b = 6*j, 6*(j + 1)
        for i in range(6):
            i2 = (i + 1) % 6
            f += [[a + i, a + i2, b + i2], [a + i, b + i2, b + i]]
            kind += [names[i]]*2
    for a in (0, 6*(len(sections) - 1)):                             # end caps
        f += [[a, a + i, a + i + 1] for i in range(1, 5)]
        kind += ['end']*4
    return v, np.array(f), np.array(kind)


# ---- the car ----
PAINT, GLASS, TYRE, RIM = rgba(.86, .2, .12), rgba(.1, .13, .18), rgba(.09, .09, .1), rgba(.75, .76, .78)
WHEEL_R, WHEEL_W = .33, .22
LF, LR = 1.156, 1.423                                               # axles (car.py); body origin = centre of gravity


def car_xml():
    rng = np.random.default_rng(0)
    mg = Merger()
    out = np.array([0., 0., .7])
    # lower body: bumper to bumper, beveled top edges (x, half-width, bottom, top, bevel)
    body = [(2.3, .78, .3, .62, .12), (2.18, .86, .24, .78, .14), (1.2, .9, .22, .86, .12), (-1.5, .9, .22, .92, .12),
            (-2.15, .87, .24, .9, .14), (-2.3, .8, .3, .72, .12)]
    v, f, kind = loft(body)
    mg.add(f'material="paint"', v, f[kind != 'bottom'], out)
    mg.add(rgba(.08, .08, .09), v, f[kind == 'bottom'], out)
    # cabin: windscreen, roof, rear window; glass on the sides and slopes, paint on the roof
    cab = [(.95, .84, .8, .9, .02), (.05, .72, .8, 1.36, .1), (-1.0, .72, .8, 1.36, .1), (-1.72, .84, .8, .94, .02)]
    v, f, kind = loft(cab)
    face_x = v[f].mean(1)[:, 0]
    roof = (kind == 'top') & (face_x < .05) & (face_x > -1.0)
    mg.add(f'material="paint"', v, f[roof | (kind == 'end')], out)
    mg.add(f'material="glass"', v, f[~roof & (kind != 'end') & (kind != 'bottom')], out)
    # lights and grille
    for side in (1, -1):
        mg.add(rgba(1, .97, .85), *box(.03, .2, .06, [2.29, side*.55, .66]), out)                 # headlights
        mg.add(rgba(.75, .05, .05), *box(.03, .22, .06, [-2.31, side*.55, .78]), out)
    mg.add(rgba(.12, .12, .13), *box(.03, .42, .07, [2.31, 0, .5]), out)                          # grille
    for xb in (2.33, -2.33):                                                                    # bumpers
        mg.add(rgba(.16, .16, .17), *box(.04, .8, .07, [xb, 0, .36]), out)
    mg.add(rgba(.92, .92, .88), *box(.02, .26, .07, [-2.33, 0, .55]), out)                          # number plate
    for x0 in (-LR,):                                                                           # rear wheels
        for side in (1, -1):
            for col, vf in wheel(side):
                v, f = vf
                mg.add(col, v + [x0, 0, 0], f, np.array([x0, side*.5, WHEEL_R]))
    assets, geoms = mg.xml('car')
    fw_assets = ''
    wheels = ''
    for k, side in enumerate((1, -1)):                                # front wheels: own bodies, they steer
        w = Merger()
        for col, (v, f) in wheel(side):
            w.add(col, v, f, np.array([0, side*.5, WHEEL_R]))
        a, g = w.xml(f'fw{k}_')
        fw_assets += a
        wheels += f'<body name="wheel{k}" mocap="true">{g}</body>'
    return assets + fw_assets, f'<body name="car" mocap="true">{geoms}</body>{wheels}'


def box(hx, hy, hz, c):
    v = np.array([[sx*hx, sy*hy, sz*hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) + c
    f = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
         [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    return v, np.array(f)


def wheel(side):
    """Tyre (10-sided) and rim (a hexagon on the outer face), axle along y at height WHEEL_R, centred at y = side*.82."""
    yc = side*.82
    def ring(k, r, y0, y1):
        v, f = prism(k, r, y0, y1)
        return v[:, [0, 2, 1]] + [0, 0, WHEEL_R], f                  # column along y
    tyre = ring(10, WHEEL_R, yc - WHEEL_W/2, yc + WHEEL_W/2)
    rim = ring(6, .2, yc + side*WHEEL_W/2 - .02, yc + side*(WHEEL_W/2 + .015))
    return [(TYRE, tyre), (RIM, rim)]


# ---- the world ----
GRASS = [(.52, .74, .34), (.48, .7, .31), (.56, .77, .37), (.45, .67, .3)]
PINE = [(.13, .38, .2), (.16, .44, .24), (.1, .33, .18)]
LEAF = [(.3, .58, .24), (.38, .64, .26), (.25, .5, .2)]
AUTUMN = [(.9, .55, .15), (.85, .35, .12), (.95, .75, .2)]
BIRCH_LEAF = [(.55, .72, .3), (.62, .76, .34)]


class Scene:
    def __init__(self, road, seed=0, w=W, h=H):
        rng = np.random.default_rng(seed)
        mg = Merger()
        x, y, yaw = road.x, road.y, road.yaw
        # road: asphalt, white edge lines, gravel shoulders
        mg.add(rgba(.34, .34, .36), *strip(x, y, yaw, -HALF_WIDTH - .3, HALF_WIDTH + .3, .02))
        for side in (-1, 1):
            mg.add(rgba(.95, .95, .92), *strip(x, y, yaw, side*HALF_WIDTH - .12, side*HALF_WIDTH + .12, .04))
            e0, e1 = sorted((side*(HALF_WIDTH + .3), side*(HALF_WIDTH + .9)))
            mg.add(rgba(.72, .66, .52), *strip(x, y, yaw, e0, e1, .01))
        # terrain: noise heights on a grid, flat near the road, growing into mountains far from it
        tree = cKDTree(np.c_[x, y])
        x0, x1, y0, y1 = x.min() - MARGIN, x.max() + MARGIN, y.min() - MARGIN, y.max() + MARGIN
        gx, gy = np.arange(x0, x1 + CELL, CELL), np.arange(y0, y1 + CELL, CELL)
        X, Y = np.meshgrid(gx, gy, indexing='ij')
        d = tree.query(np.c_[X.ravel(), Y.ravel()])[0].reshape(X.shape)
        noise = sum(zoom(rng.standard_normal((len(gx)//k + 2, len(gy)//k + 2)), k, order=3)[:len(gx), :len(gy)]*a
                    for k, a in ((40, 1.), (12, .45), (4, .15)))
        rise = np.clip((d - FLAT)/150, 0, 1)**1.5                    # 0 near the road, 1 beyond 175 m
        Z = rise*(8 + .09*d)*(1.2 + noise) - .3
        Z = np.where(d < FLAT, -.3, np.maximum(Z, -.3))
        self.Z, self.gx, self.gy = Z, gx, gy
        i, j = np.meshgrid(np.arange(len(gx) - 1), np.arange(len(gy) - 1), indexing='ij')
        i, j = i.ravel(), j.ravel()
        P = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
        idx = lambda a, b: a*len(gy) + b
        F = np.r_[np.c_[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)], np.c_[idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]]
        tri = P[F]
        zc = tri[:, :, 2].mean(1)
        steep = np.abs(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]))
        steep = 1 - steep[:, 2]/np.linalg.norm(steep, axis=1)
        band = np.where(zc > 95 + 15*rng.standard_normal(len(zc)), 3,
                        np.where((steep > .25) | (zc > 60), 2, np.where(zc > 25, 1, 0)))
        shade = rng.integers(0, 4, len(F))
        for b, cols in enumerate((GRASS, [(.4, .62, .3), (.36, .58, .28)], [(.5, .48, .45), (.45, .43, .41)],
                                  [(.93, .94, .96), (.88, .9, .93)])):
            for c, col in enumerate(cols):
                sel = (band == b) & (shade % len(cols) == c)
                if sel.any():
                    mg.add(rgba(*col), P, F[sel])
        # vegetation in clusters (forests) plus some scattered; never within 7 m of the road
        def beside(n, d0, d1):                                       # n points at d0-d1 m to the side of the road
            i = rng.integers(0, len(x), n)
            off = rng.choice([-1, 1], n)*rng.uniform(d0, d1, n)
            return np.c_[x[i] - np.sin(yaw[i])*off, y[i] + np.cos(yaw[i])*off]
        nc = len(x)//25
        centres = beside(nc, 20, 350)
        pts = np.r_[centres[rng.integers(0, nc, 12*nc)] + rng.normal(0, 30, (12*nc, 2)), beside(len(x)//4, 9, 150)]
        near = tree.query(pts)[0]
        pts = pts[near > HALF_WIDTH + 4]
        for px, py in pts:
            z = self.height(px, py) - .2
            sc, turn, u = rng.uniform(.7, 1.4), rng.uniform(0, 2*np.pi), rng.random()
            p = [px, py, z]
            if u < .4:                                               # pine: trunk and two or three stacked cones
                mg.add(rgba(.4, .27, .17), *placed(prism(5, .25*sc, 0, 1.5*sc), p, turn))
                col = rgba(*PINE[rng.integers(3)])
                for t in range(rng.integers(2, 4)):
                    mg.add(col, *placed(cone(7, (1.9 - .45*t)*sc, 2.4*sc, (1.1 + 1.3*t)*sc), p, turn))
            elif u < .7:                                             # round tree, some in autumn colours
                mg.add(rgba(.42, .3, .2), *placed(prism(5, .2*sc, 0, 2.2*sc, .14*sc), p, turn))
                col = AUTUMN[rng.integers(3)] if rng.random() < .18 else LEAF[rng.integers(3)]
                mg.add(rgba(*col), *placed(blob(rng, 1.8*sc, 1.8*sc, 1.6*sc, 3.3*sc), p, turn), np.array([px, py, z + 3.3*sc]))
            elif u < .8:                                             # birch: thin white trunk, small light crown
                mg.add(rgba(.92, .9, .86), *placed(prism(5, .12*sc, 0, 3.5*sc, .08*sc), p, turn))
                mg.add(rgba(*BIRCH_LEAF[rng.integers(2)]), *placed(blob(rng, .9*sc, .9*sc, 1.6*sc, 4*sc), p, turn),
                       np.array([px, py, z + 4*sc]))
            elif u < .93:                                            # bush
                mg.add(rgba(*LEAF[rng.integers(3)]), *placed(blob(rng, .9*sc, .9*sc, .6*sc, .3), p, turn),
                       np.array([px, py, z + .3]))
            else:                                                    # rock
                mg.add(rgba(*[(.55, .55, .57), (.5, .49, .47)][rng.integers(2)]),
                       *placed(blob(rng, 1.2*sc, 1.*sc, .7*sc, 0., .3), p, turn), np.array([px, py, z]))
        assets, geoms = mg.xml()
        ca, cg = car_xml()
        mx, my = x.mean(), y.mean()
        xml = f'''<mujoco><statistic extent="100" center="{mx:.0f} {my:.0f} 0"/>
<visual><global offwidth="{w}" offheight="{h}"/><quality shadowsize="4096" offsamples="4"/>
 <map fogstart="2.5" fogend="13" znear=".002" zfar="45" shadowclip=".5"/><rgba fog=".74 .83 .93 1"/>
 <headlight ambient=".5 .52 .56" diffuse=".2 .2 .2" specular="0 0 0"/></visual>
<asset><texture name="sky" type="skybox" builtin="gradient" rgb1=".38 .6 .92" rgb2=".8 .88 .97" width="256" height="256"/>
 <material name="paint" rgba=".86 .2 .12 1" specular=".5" shininess=".7" reflectance="0"/>
 <material name="glass" rgba=".1 .13 .18 1" specular=".8" shininess=".9"/>{assets}{ca}</asset>
<worldbody>{geoms}{cg}
 <body name="sun" mocap="true"><light name="sun" directional="true" castshadow="true" dir="-.45 .35 -.8"
  diffuse=".85 .8 .7" pos="0 0 80"/></body></worldbody></mujoco>'''
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, h, w)
        self.cam = mujoco.MjvCamera()
        self.cam.distance, self.cam.elevation = 8.5, -13.
        self.view_yaw = None
        self.wheel_ids = [self.m.body(f'wheel{k}').mocapid[0] for k in range(2)]
        self.car_id, self.sun_id = self.m.body('car').mocapid[0], self.m.body('sun').mocapid[0]

    def height(self, px, py):
        """Terrain height at (px, py), bilinear on the grid."""
        fi, fj = (px - self.gx[0])/CELL, (py - self.gy[0])/CELL
        i, j = int(fi), int(fj)
        a, b = fi - i, fj - j
        Z = self.Z
        return (1 - a)*(1 - b)*Z[i, j] + a*(1 - b)*Z[i + 1, j] + (1 - a)*b*Z[i, j + 1] + a*b*Z[i + 1, j + 1]

    def render(self, x):
        """RGB frame of car state x (7,) (numpy or torch), from a camera behind it."""
        X, Y, yaw, delta = float(x[0]), float(x[1]), float(x[2]), float(x[6])
        c, s = np.cos(yaw), np.sin(yaw)
        quat = lambda a: [np.cos(a/2), 0, 0, np.sin(a/2)]
        self.d.mocap_pos[self.car_id], self.d.mocap_quat[self.car_id] = [X, Y, 0], quat(yaw)
        self.d.mocap_pos[self.sun_id] = [X, Y, 0]                     # the shadow box follows the car
        for k in range(2):                                            # front wheels turn with the steering
            self.d.mocap_pos[self.wheel_ids[k]] = [X + LF*c, Y + LF*s, 0]
            self.d.mocap_quat[self.wheel_ids[k]] = quat(yaw + delta)
        mujoco.mj_kinematics(self.m, self.d)
        # camera trails the car's heading with a short lag, so it does not shake with every steering correction
        if self.view_yaw is None:
            self.view_yaw = yaw
        self.view_yaw += .25*np.arctan2(np.sin(yaw - self.view_yaw), np.cos(yaw - self.view_yaw))
        self.cam.lookat[:] = [X, Y, 1.3]
        self.cam.azimuth = np.degrees(self.view_yaw)
        self.r.update_scene(self.d, self.cam)
        return self.r.render()
