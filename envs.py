import numpy as np

import fym.core as core
import fym.agents.LQR as LQR


class Base(core.BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_inner_ctrl(self, inner_ctrl):
        self.inner_ctrl = inner_ctrl
        self.get_action = inner_ctrl.get


class Agent(core.BaseEnv):
    def __init__(self, w1, w2, w3, eta=1e-2, R=1):
        super().__init__()
        self.w1 = core.BaseSystem(w1)
        self.w2 = core.BaseSystem(w2)
        self.w3 = core.BaseSystem(w3)
        self.R = R
        self.eta = eta

    def get(self, t, x):
        return self.phi2(x).T.dot(self.w2.state)

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

        agent = Agent(
            w1=np.zeros_like(self.true_w1),
            w2=np.zeros_like(self.true_w2),
            w3=np.zeros_like(self.true_w3),
            R=self.R,
        )
        self.set_inner_ctrl(agent)

    def get_true_parameters(self):
        # Construct the true Hamiltonian
        K, P = LQR.clqr(self.A, self.B, self.Q, self.R)[:2]
        XX = P.dot(self.B).dot(K)
        w1 = XX[np.triu_indices(XX.shape[0])]
        XU = 2 * P.dot(self.B)
        w1 = np.hstack((w1, XU.T.ravel()))
        w1 = np.hstack((w1, self.R[np.triu_indices(self.R.shape[0])]))

        # phi1 = x1**2, x1*x2, x1*x3, x2**2, x2*X3, x3**2, x1*u, x2*u, x3*u, u**2
        # phi2 = x1, x2, x3
        # phi3 = x1, x2, x3
        self.true_w1 = w1[:, None]
        self.true_w2 = K.ravel()[:, None]
        self.true_w3 = 4 * self.B.T.dot(P).ravel()[:, None]

    def set_dot(self, t):
        x = self.main.state
        u = self.get_action(t, x)
        self.main.dot = self.A.dot(x) + self.B.dot(u)
        self.inner_ctrl.set_dot(x, u)

    def reset(self, mode="normal"):
        super().reset()
        if mode == "random":
            self.main.initial_state = (
                self.main.initial_state
                + np.random.randn(*self.main.state_shape)
            )


if __name__ == "__main__":
    env = F16Dof3()
    env.reset(mode="random")
    env.set_dot(0)
