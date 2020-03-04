import copy
import numpy as np

import fym.core as core
import fym.agents.LQR as LQR

import common


class Base(core.BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_inner_ctrl(self, inner_ctrl):
        self.inner_ctrl = inner_ctrl

    def get_name(self):
        return '-'.join([self.name, self.agent.name])


class BaseInnerCtrl:
    def __init__(self, xdim, udim):
        self.xtrim = np.zeros((xdim, 1))
        self.utrim = np.zeros((udim, 1))

    def set_trim(self, xtrim, utrim):
        self.xtrim = xtrim
        self.utrim = utrim

    def get(self, t, x):
        raise NotImplementedError


class Linear(BaseInnerCtrl):
    def __init__(self, xdim, udim):
        super().__init__(xdim, udim)
        self.theta = np.zeros((xdim, udim))
        self.noise = []

    def get_param(self):
        return self.theta

    def set_param(self, theta):
        if np.ndim(theta) == 0:
            theta = theta * np.ones_like(self.theta)

        assert np.ndim(theta) == np.ndim(self.theta)

        self.theta = theta

    def set_phi(self, func):
        self.phi = func

    def get(self, t, x):
        theta = self.theta
        return self.phi(x).T.dot(theta) + self.get_noise(t)

    def get_noise(self, t):
        res = 0
        if self.noise:
            for noise in self.noise:
                res = res + noise(t)
        return res

    def add_noise(self, noise):
        self.noise.append(noise)


class Agent(core.BaseEnv):
    def __init__(self, w1, w2, w3, eta=1e3, R=1):
        super().__init__()
        self.w1 = core.BaseSystem(w1)
        self.w2 = core.BaseSystem(w2)
        self.w3 = core.BaseSystem(w3)
        self.R = R
        self.eta = eta

    def get(self, t, x):
        return self.phi2(x).T.dot(self.w2.state)

    def get_param(self):
        return self.w2.state.copy()

    def set_dot(self, x, u):
        w1, w2, w3 = self.w1.state, self.w2.state, self.w3.state
        phi1, phi2, phi3 = self.phi1(x, u), self.phi2(x), self.phi3(x)
        udiff = u - phi2.T.dot(w2)
        e = (
            phi1.T.dot(w1)
            - udiff.T.dot(self.R).dot(udiff)
            - udiff.T.dot(phi3.T.dot(w3))
        )
        self.w1.dot = - self.eta * e * phi1
        self.w2.dot = - self.eta * e * (
            2 * phi2.dot(self.R).dot(udiff)
            + phi2.dot(phi3.T).dot(w3)
        )
        self.w3.dot = self.eta * e * phi3.dot(udiff)

    def phi1(self, x, u):
        # x1**2, x1*x2, x1*x3, x2**2, x2*X3, x3**2, x1*u, x2*u, x3*u, u**2
        phi = np.vstack((x[0] * x, x[1] * x[1:], x[2] * x[2:], x * u, u ** 2))
        return phi

    def phi2(self, x):
        return x

    def phi3(self, x):
        return x


class F16Dof3(Base):
    """Examples are from
    Yuanheng Zhu and Dongbin Zhao,
    "Comprehensive comparison of online ADP algorithms
    for continuous-time optimal control,"
    Artif Intell Rev, 49:531-547, 2018
    doi: https://doi.org/10.1007/s10462-017-9548-4
    """
    A = np.array([
        [-1.01887, 0.90506, -0.00215],
        [0.82225, -1.07741, -0.17555],
        [0, 0, -1]])
    B = np.array([[0], [0], [1]])
    Q = np.diag(np.ones(3))
    R = np.diag(np.ones(1))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.main = core.BaseSystem(np.zeros(3)[:, None])
        self.get_true_parameters()

        self.agent = Agent(
            w1=np.random.random(self.true_w1.shape),
            w2=np.zeros_like(self.true_w2),
            w3=np.random.random(self.true_w3.shape),
            R=self.R,
        )

        behavior = Linear(xdim=3, udim=1)
        behavior.set_phi(self.agent.phi2)
        # ou_noise = common.OuNoise(
        #     0, 0.5, dt=1, max_t=kwargs["max_t"], decay=kwargs["max_t"])
        # behavior.add_noise(ou_noise)
        # behavior.add_noise(lambda t: np.sin(t / 10))
        self.set_inner_ctrl(behavior)

        self.set_logger_callback()

    def set_logger_callback(self, func=None):
        self.logger_callback = func or self.get_info

    def get_info(self, i, t, y, thist, odehist):
        x, inner = self.observe_dict(y).values()
        u = self.inner_ctrl.get(t, x)
        us = self.agent.get(t, x)
        return {
            "time": t,
            "state": x,
            "action": u,
            "target_action": us,
            "agent_state": inner
        }

    def get_true_parameters(self):
        # Construct the true Hamiltonian
        K, P = LQR.clqr(self.A, self.B, self.Q, self.R)[:2]
        XX = P.dot(self.B).dot(K)
        XX = 2 * XX - np.diag(XX)
        w1 = XX[np.triu_indices(XX.shape[0])]
        XU = 2 * P.dot(self.B)
        w1 = np.hstack((w1, XU.T.ravel()))
        w1 = np.hstack((w1, self.R[np.triu_indices(self.R.shape[0])]))

        # phi1 = x1**2, x1*x2, x1*x3, x2**2, x2*X3, x3**2, x1*u, x2*u, x3*u, u**2
        # phi2 = x1, x2, x3
        # phi3 = x1, x2, x3
        self.true_w1 = w1[:, None]
        self.true_w2 = - K.ravel()[:, None]
        self.true_w3 = 4 * self.B.T.dot(P).ravel()[:, None]

    def step(self):
        self.update()
        done = self.clock.time_over()
        return None, None, done, None

    def set_dot(self, t):
        x = self.main.state

        self.inner_ctrl.set_param(self.agent.get_param())
        u = self.inner_ctrl.get(t, x)

        self.main.dot = (self.A + 0.1 * np.eye(3)).dot(x) + self.B.dot(u)
        self.agent.set_dot(x, u)

    def reset(self, mode="normal"):
        super().reset()
        if mode == "random":
            self.main.initial_state = (
                self.main.initial_state
                + np.random.randn(*self.main.state_shape)
            )


if __name__ == "__main__":
    env = F16Dof3(max_t=10)
    env.reset(mode="random")
    env.set_dot(0)
