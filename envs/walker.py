"""Planar walker (DeepMind Control Suite walker-walk) in plain MuJoCo, with motor faults.

Matches dm_control's task: the same model, 0.025 s control steps (10 physics steps), observation
[body orientations (14), torso height (1), joint velocities (9)] and walking reward. Each decision repeats the
command for 2 control steps (as TD-MPC2 was trained), so rewards are in [0, 2].

A motor fault makes the motors deliver a = clip(gain * c + offset, -1, 1) for the command c.
"""

from pathlib import Path

import mujoco
import numpy as np

XML = Path(__file__).with_name("walker.xml")
MOTORS = ("right hip", "right knee", "right ankle", "left hip", "left knee", "left ankle")
STAND_HEIGHT, WALK_SPEED = 1.2, 1.
SUBSTEPS, ACTION_REPEAT = 10, 2   # physics steps per control step; control steps per decision


def tolerance(x, lower, margin, value_at_margin=.1, sigmoid="gaussian"):
    """dm_control's rewards.tolerance with bounds (lower, inf)."""
    d = np.maximum(lower - x, 0) / margin
    if sigmoid == "gaussian":
        return np.exp(-.5 * (d * np.sqrt(-2 * np.log(value_at_margin)))**2)
    scaled = d * (1 - value_at_margin)  # linear
    return np.where(scaled < 1, 1 - scaled, 0.)


class WalkerEnv:
    def __init__(self, seed=0):
        self.model = mujoco.MjModel.from_xml_path(str(XML))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.gain, self.offset = np.ones(6), np.zeros(6)
        self.renderer = self.camera = None

    def set_fault(self, gain=None, offset=None):
        """Motor i delivers clip(gain[i] * c[i] + offset[i]); no arguments restores healthy motors."""
        self.gain = np.ones(6) if gain is None else np.asarray(gain, float)
        self.offset = np.zeros(6) if offset is None else np.asarray(offset, float)

    def reset(self):
        """Random joint angles and body pitch, 1.3 m up (as in dm_control): the walker falls in its first steps."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                low, high = m.jnt_range[j] if m.jnt_limited[j] else (-np.pi, np.pi)
                d.qpos[m.jnt_qposadr[j]] = self.rng.uniform(low, high)
        mujoco.mj_forward(m, d)
        return self.observation()

    def step(self, command):
        """Apply the command (6 numbers in [-1, 1]) for one decision; returns (observation, reward)."""
        self.data.ctrl[:] = np.clip(self.gain * np.asarray(command) + self.offset, -1, 1)
        reward = 0.
        for _ in range(ACTION_REPEAT):
            mujoco.mj_step2(self.model, self.data)  # dm_control's step order: sensors match the new state
            mujoco.mj_step(self.model, self.data, SUBSTEPS - 1)
            mujoco.mj_step1(self.model, self.data)
            reward += self.reward()
        return self.observation(), reward

    def observation(self):
        d = self.data
        orientations = d.xmat[1:][:, [0, 2]].ravel()  # xx, xz of each body
        return np.concatenate([orientations, [d.xpos[1, 2]], d.qvel]).astype(np.float32)

    def reward(self):
        d = self.data
        standing = tolerance(d.xpos[1, 2], STAND_HEIGHT, STAND_HEIGHT / 2)
        upright = (1 + d.xmat[1, 8]) / 2
        move = tolerance(d.sensordata[0], WALK_SPEED, WALK_SPEED / 2, .5, "linear")
        return float((3 * standing + upright) / 4 * (5 * move + 1) / 6)

    def render(self, height=480, width=640):
        """RGB image from a camera following the torso from the side."""
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height, width)
            self.camera = mujoco.MjvCamera()
            self.camera.distance, self.camera.azimuth, self.camera.elevation = 4., 90., -10.
        self.camera.lookat[:] = self.data.xpos[1, 0], 0., 1.
        self.renderer.update_scene(self.data, self.camera)
        return self.renderer.render()

    def copy_state_from(self, other):
        """Continue from another walker's state (used to start the frozen agent where the adaptive one is)."""
        self.data.qpos[:], self.data.qvel[:], self.data.time = other.data.qpos, other.data.qvel, other.data.time
        mujoco.mj_forward(self.model, self.data)
