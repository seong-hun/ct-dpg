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


class WSystem(core.BaseEnv):
    def __init__(self, x_size, u_size, get_all_basis):
        super().__init__()

        x = np.zeros((x_size, 1))
        u = np.zeros((u_size, 1))
        basis = get_all_basis(x, u)
        self.w1 = core.BaseSystem(np.zeros_like(basis[0]))
        self.w2 = core.BaseSystem(np.zeros_like(basis[1]))
        self.w3 = core.BaseSystem(np.zeros_like(basis[2]))
        self.w2p = core.BaseSystem(np.zeros_like(basis[3]))
        self.w3p = core.BaseSystem(np.zeros_like(basis[4]))


class Agent(core.BaseEnv):
    def __init__(self, x_size, u_size, k, theta, eta1=1e3, eta2=1e-1, R=1):
        super().__init__()

        self.R = R
        self.k = k
        self.kdiff = np.diff(self.k)
        self.theta = theta
        self.eta1 = eta1
        self.eta2 = eta2

        self.wsys = WSystem(x_size, u_size, self.get_all_basis)
        shape = self.wsys.state[:, None].shape
        self.M = core.BaseSystem(np.zeros((shape[0], shape[0])))
        self.N = core.BaseSystem(np.zeros(shape))

    def get(self, t, x, w2=None):
        if w2 is None:
            w2 = self.w2.state
        return self.phi2(x).T.dot(w2)

    def get_param(self):
        return self.wsys.w2.state.copy()

    def get_k(self, e):
        return self.k[0] + self.kdiff * np.tanh(self.theta * np.abs(e))

    def set_dot(self, t, x, u):
        M, N = self.M.state, self.N.state

        w = self.wsys.state[:, None]
        basis = np.vstack(self.get_all_basis(x, u))
        y = u.T.dot(self.R).dot(u)
        e = basis.T.dot(w) - y

        if t == 0:
            self.ta, self.Ma, self.Na = t, M, N
        elif t > self.ta:
            if get_eigvals(M).min() >= get_eigvals(self.Ma).min():
                self.ta, self.Ma, self.Na = t, M, N

        k = self.get_k(e)
        Ma, Na = self.Ma, self.Na

        wdot = -self.eta1 * e * basis - self.eta2 * (Ma.dot(w) - Na)
        self.wsys.dot = wdot.ravel()

        # Additional
        _, w2, w3, w2p, w3p = self.wsys.observe_list()
        phi3 = self.phi3(x)
        e2 = np.kron(w2, w2) - w2p
        e3 = np.kron(w2, w3) - w3p

        self.wsys.w2.dot += -self.eta1 * (
            np.kron(np.eye(w2.size), w2).T.dot(e2)
        )
        self.wsys.w2p.dot += self.eta1 * (
            np.eye(w2p.size).dot(e2)
        )

        self.wsys.w2.dot += -self.eta1 * (
            np.kron(np.eye(w2.size), w3).T.dot(e3)
        )
        self.wsys.w3.dot += -self.eta1 * (
            np.kron(w2, np.eye(w3.size)).T.dot(e3)
        )
        self.wsys.w3p.dot += self.eta1 * (
            np.eye(w3p.size).dot(e3)
        )

        self.wsys.w3.dot += -self.eta1 * phi3.dot(phi3.T).dot(w3)

        norm = basis.T.dot(basis) + 1
        self.M.dot = - k * M + basis.dot(basis.T) / norm
        self.N.dot = - k * N + basis.dot(y.T) / norm

    def phi1(self, x, u):
        # x1**2, x1*x2, x1*x3, x2**2, x2*X3, x3**2, x1*u, x2*u, x3*u, u**2
        phi = np.vstack((x[0] * x, x[1] * x[1:], x[2] * x[2:], x * u, u ** 2))
        return phi

    def phi2(self, x):
        return x

    def phi3(self, x):
        return x

    def get_all_basis(self, x, u):
        phi2 = self.phi2(x)
        phi3 = self.phi3(x)
        R = self.R
        return (
            self.phi1(x, u),
            2 * phi2.dot(R).dot(u),
            -phi3.dot(u),
            vec(phi2.dot(R).dot(phi2.T)),
            vec(phi2.dot(phi3.T))
        )


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
    Q = np.diag(np.ones(3))
    R = np.diag(np.ones(1))

    def __init__(self, eta1=1e1, eta2=0e1, k=(0.1, 10), theta=0.1,
                 turnon=50, **kwargs):
        super().__init__(**kwargs)
        self.main = core.BaseSystem(np.zeros(3)[:, None])
        self.get_true_parameters()

        self.agent = Agent(
            x_size=3, u_size=1, k=k, theta=theta, R=self.R,
            eta1=eta1, eta2=eta2,
        )

        behavior = Linear(xdim=3, udim=1, turnon=turnon)
        behavior.set_phi(self.agent.phi2)
        noise = common.OuNoise(
            0, 0.5, dt=0.5, max_t=kwargs["max_t"], decay=kwargs["max_t"])
        behavior.add_noise(noise)
        behavior.add_noise(lambda t: np.sin(t) + np.cos(1.3*t))
        self.set_inner_ctrl(behavior)

        self.set_logger_callback()

    def set_logger_callback(self, func=None):
        self.logger_callback = func or self.get_info

    def get_info(self, i, t, y, thist, odehist):
        x, agent, inner_ctrl = self.observe_dict(y).values()
        w2 = agent["wsys"]["w2"]
        ut = self.agent.get(t, x, w2)
        u = self.inner_ctrl.get(t, x, w2)
        return {
            "time": t,
            "state": x,
            "action": u,
            "target_action": ut,
            "agent_state": agent
        }

    def step(self):
        self.update()
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
        K, P = LQR.clqr(self.A, self.B, self.Q, self.R)[:2]
        XX = P.dot(self.B).dot(K)
        XX = 2 * XX - np.diag(XX)
        w1 = XX[np.triu_indices(XX.shape[0])]
        XU = 2 * P.dot(self.B)
        w1 = np.hstack((w1, XU.T.ravel()))
        w1 = np.hstack((w1, self.R[np.triu_indices(self.R.shape[0])]))

        self.true_w1 = w1[:, None]
        self.true_w2 = - K.ravel()[:, None]


def vec(X):
    return X.reshape(-1, 1, order="F")


def get_eigvals(M):
    eigvals = np.linalg.eigvals(M)
    eigvals[np.isclose(eigvals, 0)] = 0
    return eigvals


if __name__ == "__main__":
    env = F16Dof3(max_t=10)
    env.reset(mode="random")
    env.set_dot(0)
