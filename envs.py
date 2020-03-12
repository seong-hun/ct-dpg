import copy
import numpy as np

import fym.core as core
import fym.agents.LQR as LQR

import common


class BaseInnerCtrl(core.BaseEnv):
    def __init__(self, xdim, udim):
        super().__init__()
        self.xtrim = np.zeros((xdim, 1))
        self.utrim = np.zeros((udim, 1))
        self.noise = []

    def set_trim(self, xtrim, utrim):
        self.xtrim = xtrim
        self.utrim = utrim

    def get(self, t, x):
        raise NotImplementedError

    def get_noise(self, t):
        res = 0
        if self.noise:
            for noise in self.noise:
                res = res + noise(t)
        return res

    def add_noise(self, noise):
        self.noise.append(noise)


class Linear(BaseInnerCtrl):
    def __init__(self, xdim, udim, turnon):
        super().__init__(xdim, udim)
        self.model = core.BaseSystem(np.zeros((xdim, udim)))
        self.turnon = turnon

    def get_param(self):
        return self.model.state.copy()

    def set_param(self, state):
        if np.ndim(state) == 0:
            state = state * np.ones_like(self.model.state)

        assert np.ndim(state) == np.ndim(self.model.state)
        self.model.state = state

    def set_phi(self, func):
        self.phi = func

    def get(self, t, x, state=None):
        if state is None:
            state = self.model.state

        ut = self.phi(x).T.dot(state)
        if t < self.turnon:
            ut = np.zeros_like(ut) + self.get_noise(t)

        return ut


class Behavior(BaseInnerCtrl):
    def __init__(self, xdim, udim, eta2=1):
        super().__init__(xdim, udim)
        self.model = core.BaseSystem(np.random.randn(udim, 1))
        self.eta2 = eta2

    def get(self, t, x, u=None):
        if u is None:
            u = self.model.state
        return u

    def set_dot(self, t, x, ut):
        udiff = self.model.state - ut
        self.model.dot = - self.eta2 * udiff + self.get_noise(t)


class Critic(core.BaseEnv):
    def __init__(self, x_size, u_size):
        super().__init__()

        x = np.zeros((x_size, 1))
        u = np.zeros((u_size, 1))
        self.w = core.BaseSystem(np.zeros_like(self.basis(x, u, u)))

    def phi1(self, x, u):
        # x1**2, x1*x2, x1*x3, x2**2, x2*X3, x3**2, x1*u, x2*u, x3*u, u**2
        phi = np.vstack((x[0] * x, x[1] * x[1:], x[2] * x[2:], x * u, u ** 2))
        return phi

    def grad_phi1(self, x, u):
        return np.hstack((np.zeros((1, 6)), x.T, 2 * u.T))

    def basis(self, x, u, pi):
        return self.phi1(x, u) - self.grad_phi1(x, u).T.dot(u - pi)


class Actor(core.BaseEnv):
    def __init__(self, x_size, u_size):
        super().__init__()
        x = np.zeros((x_size, 1))
        self.w = core.BaseSystem(
            # np.zeros((self.basis(x).shape[0], u_size)))
            np.random.random((self.basis(x).shape[0], u_size)))

    def phi2(self, x):
        return x

    def basis(self, x):
        return self.phi2(x)


class Agent(core.BaseEnv):
    Q = np.diag(np.ones(3))
    R = np.diag(np.ones(1))

    def __init__(self, x_size, u_size, k, theta, eta1=1e3, eta2=1e-1):
        super().__init__()

        self.k = k
        self.kdiff = np.diff(self.k)
        self.theta = theta
        self.eta1 = eta1
        self.eta2 = eta2

        self.critic = Critic(x_size, u_size)
        self.actor = Actor(x_size, u_size)
        shape = self.critic.w.state.shape
        self.M = core.BaseSystem(np.zeros((shape[0], shape[0])))
        self.N = core.BaseSystem(np.zeros(shape))

    def get(self, t, x, w2=None):
        if w2 is None:
            w2 = self.actor.w.state
        return self.actor.basis(x).T.dot(w2)

    def get_param(self):
        return self.actor.w.state.copy()

    def get_k(self, e):
        # print(np.tanh(self.theta * np.abs(e)))
        return self.k[0] + self.kdiff * np.tanh(self.theta * np.abs(e))

    def reward(self, x, u):
        return x.T.dot(self.Q).dot(x) + u.T.dot(self.R).dot(u)

    def grad_reward(self, x, u):
        return 2 * self.R.dot(u)

    def set_dot(self, t, x, u):
        w1 = self.critic.w.state
        w2 = self.actor.w.state
        M, N = self.M.state, self.N.state

        basis2 = self.actor.basis(x)
        pi = basis2.T.dot(w2)
        basis1 = self.critic.basis(x, u, pi)
        grad_phi1 = self.critic.grad_phi1(x, pi)

        r, rpi = self.reward(x, u), self.reward(x, pi)
        gradr = self.grad_reward(x, u)

        y = (r - rpi - (u - pi).T.dot(gradr))
        ec = w1.T.dot(basis1) - y
        self.critic.w.dot = -self.eta1 * ec * basis1 / (basis1.T.dot(basis1) + 1)
        # self.critic.w.dot += -self.eta * 1
        self.actor.w.dot = -self.eta2 * basis2.dot(grad_phi1).dot(w1)
        # self.actor.w.dot += -self.eta1 * ec * grad_phi1.dot(w1)

        if t == 0:
            self.ta, self.Ma, self.Na = t, M, N
        elif t > self.ta:
            eigM = get_eigvals(M)
            eigMa = get_eigvals(self.Ma)
            cond1 = len(eigM) > len(eigMa)
            cond2 = len(eigM) == len(eigMa) and eigM.min() >= eigMa.min()
            if cond1 or cond2:
                self.ta, self.Ma, self.Na = t, M, N
                self.Ma[np.isclose(self.Ma, 0)] = 0
                self.Na[np.isclose(self.Na, 0)] = 0

            # print(get_eigvals(M, reduce=False).min())

        k = self.get_k(ec)
        Ma, Na = self.Ma, self.Na

        self.critic.w.dot += -(Ma.dot(w1) - Na)

        norm = basis1.T.dot(basis1) + 1
        self.M.dot = - k * M + basis1.dot(basis1.T) / norm
        self.N.dot = - k * N + basis1.dot(y.T) / norm

        # if t > 1:
        #     breakpoint()

        if np.isnan(M).any() or np.isnan(N).any():
            breakpoint()

        if np.isnan(self.dot).any():
            breakpoint()


class Base(core.BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_inner_ctrl(self, inner_ctrl):
        self.inner_ctrl = inner_ctrl

    def get_name(self):
        return '-'.join([self.name, self.agent.name])


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

    def __init__(self, eta1, eta2, k, theta,
                 noise_mean, noise_std, noise_dt,
                 turnon, **kwargs):
        super().__init__(**kwargs)
        self.main = core.BaseSystem(np.zeros(3)[:, None])
        self.agent = Agent(
            x_size=3, u_size=1, k=k, theta=theta,
            eta1=eta1, eta2=eta2,
        )

        self.turnon = turnon
        self.impulse = [
            (3, np.array([1, 0, 0])[:, None]),
            (6, np.array([0.5, 1, -0.5])[:, None]),
            (9, np.array([-1, 0.5, 1])[:, None]),
            (13, np.array([2, 1.5, -1.5])[:, None]),
            # (18, np.array([1, 0, 0.5])[:, None]),
        ]

        self.get_true_parameters()

        behavior = Linear(xdim=3, udim=1, turnon=turnon)
        behavior.set_phi(self.agent.actor.basis)
        noise = common.OuNoise(
            noise_mean, noise_std, dt=noise_dt,
            max_t=kwargs["max_t"], decay=kwargs["max_t"])
        behavior.add_noise(noise)
        behavior.add_noise(lambda t: (
            0.4 * np.sin(2 * t) * (1 + 0.2 * np.cos(1.3 * t))
            + np.cos(2.5 * t + 0.1)
            - np.cos(0.5 * t + 0.12)
        ))
        self.set_inner_ctrl(behavior)

        self.set_logger_callback()

    def set_logger_callback(self, func=None):
        self.logger_callback = func or self.get_info

    def get_info(self, i, t, y, thist, odehist):
        x, agent, inner_ctrl = self.observe_dict(y).values()
        w2 = agent["actor"]["w"]
        ut = self.agent.get(t, x, w2)
        u = self.inner_ctrl.get(t, x, w2)
        eigM = get_eigvals(agent["M"], reduce=False)
        eigMa = get_eigvals(self.agent.Ma, reduce=False)
        return {
            "time": t,
            "state": x,
            "action": u,
            "target_action": ut,
            "critic_state": agent["critic"]["w"],
            "actor_state": w2,
            "eigM_min": eigM.min(),
            "eigMa_min": eigMa.min(),
        }

    def step(self):
        self.update()
        if self.impulse:
            t, state = self.impulse[0]
            if self.clock.get() > t:
                self.main.state = state
                self.impulse.pop(0)
        done = self.clock.time_over()
        return None, None, done, None

    def set_dot(self, t):
        x = self.main.state

        u = self.inner_ctrl.get(t, x, self.agent.get_param())

        self.main.dot = self.A.dot(x) + self.B.dot(u)
        self.agent.set_dot(t, x, u)
        self.inner_ctrl.dot = np.zeros_like(self.inner_ctrl.state)

    def reset(self, mode="normal"):
        super().reset()
        if mode == "random":
            self.main.initial_state = (
                self.main.initial_state
                + np.random.randn(*self.main.state_shape)
            )

    def get_true_parameters(self):
        # Construct the true Hamiltonian
        Q = self.agent.Q
        R = self.agent.R
        K, P = LQR.clqr(self.A, self.B, Q, R)[:2]
        XX = K.T.dot(R).dot(K)
        w1 = XX[np.triu_indices(XX.shape[0])]
        XU = 2 * K.T.dot(R)
        w1 = np.hstack((w1, XU.T.ravel()))
        w1 = np.hstack((w1, R[np.triu_indices(R.shape[0])]))

        self.true_w1 = w1[:, None]
        self.true_w2 = - K.ravel()[:, None]


def vec(X):
    return X.reshape(-1, 1, order="F")


def get_eigvals(M, reduce=True, symmetric=True):
    eigvals = np.linalg.eigvals(M)
    if symmetric:
        eigvals = eigvals.real
    if reduce:
        eigvals = eigvals[~np.isclose(eigvals, 0)]
    else:
        eigvals[np.isclose(eigvals, 0)] = 0
    return eigvals


if __name__ == "__main__":
    env = F16Dof3(1, 1, (1, 1), 1, 1, 1, 1, 1, max_t=10)
    env.reset(mode="random")
    env.set_dot(0)
