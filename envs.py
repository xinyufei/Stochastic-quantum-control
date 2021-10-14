import gym
import numpy as np
from abc import ABC
from scipy.linalg import expm


class QuantumSystem(gym.Env, ABC):
    def __init__(self, _=None, d_hamil=[], c_hamil=None, init_state=None, targ_state=None, time_steps=400, evo_time=20,
                 obj_mode='fid'):
        self.d_hamil = d_hamil
        self.c_hamil = c_hamil
        self.init_state = init_state
        self.targ_state = targ_state
        self.time_steps = time_steps
        self.evo_time = evo_time
        self.obj_mode = obj_mode

        self.num_ctrl = len(d_hamil)
        self.cur_state = init_state
        self.delta_t = self.evo_time / self.time_steps

        self._in_evo = False

        self._init_action_observation_space()

    def step(self, action):
        if not self._in_evo:
            # start the evolution
            self.cur_state = self.init_state
            self._in_evo = True

        # conduct time evolution
        self._apply_control(action)

        # compute the reward





    def _init_action_observation_space(self):
        self.observation_space = gym.spaces.Box(low=-np.infty, high=-np.infty, shape=self.init_state.shape)
        self.action_space = gym.spaces.MultiDiscrete(self.num_ctrl)

    def _apply_control(self, action):
        hamil = self.d_hamil.copy() + self.c_hamil[action]
        new_state = expm(-1j * hamil * self.delta_t).dot(self.cur_state)
        self.cur_state = new_state.copy()

