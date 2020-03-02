import numpy as np
import numba as nb

import fym.core as core
import fym.agents.LQR as LQR
from fym.models.aircraft import F16LinearLateral


@nb.njit()
def quad_reward(x_trimmed, u_trimmed, Q, R):
    return (np.sum(np.dot(x_trimmed, Q) * x_trimmed, axis=-1)
            + np.sum(np.dot(u_trimmed, R) * u_trimmed, axis=-1))


@nb.njit()
def quad_reward_grad(u_trimmed, R):
    return 2 * np.dot(u_trimmed, R)


class F16(core.BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system = F16LinearLateral()
        self.logger_callback = self.get_info

        self.n = self.system.B.shape[0]
        self.m = self.system.B.shape[1]
        self.Q = np.diag(np.ones(self.n)).astype(np.float32)
        self.R = np.diag(np.ones(self.m)).astype(np.float32)

        self.K = -LQR.clqr(self.system.A, self.system.B, self.Q, self.R)[0]

    def reset(self, mode="normal"):
        super().reset()
        if mode == "random":
            self.system.initial_state = (
                self.system.initial_state
                + np.random.randn(*self.system.state_shape)
            )
            self.K = self.K * (1 + 0.05 * np.random.randn(*self.K.shape))

    def step(self):
        _, _, eager_done = self.update()
        done = self.clock.time_over() or eager_done
        return None, None, done, {}

    def set_dot(self, time):
        x = self.system.state
        u = self.get_action(time, x)
        self.system.dot = self.system.deriv(x, u)

    def get_eigvals(self, param):
        Ac = self.system.A + self.system.B.dot(param)
        return np.linalg.eigvals(Ac)

    def get_info(self, i, t, y, t_hist, ode_hist):
        ny = ode_hist[i + 1]
        x, nx = [
            p[system.flat_index].reshape(system.state_shape)
            for p in (y, ny)
            for system in self.systems
        ]
        u = self.get_action(t, x)
        reward = self.reward(x, u)
        return {
            "time": t,
            "state": x,
            "action": u,
            "reward": reward,
            "next_state": nx
        }

    def get_action(self, t, x):
        return self.K.dot(x)

    def set_inner_ctrl(self, inner_ctrl):
        self.inner_ctrl = inner_ctrl
        self.get_action = inner_ctrl.get

    def get_param(self):
        if hasattr(self, "inner_ctrl"):
            return self.inner_ctrl.get_param()
        else:
            return self.K

    def reward(self, x, u):
        x = x.astype(np.float32)
        u = u.astype(np.float32)
        return quad_reward(x, u, self.Q, self.R)

    def reward_grad(self, x, u):
        return quad_reward_grad(u, self.R)


if __name__ == "__main__":
    env = F16()
