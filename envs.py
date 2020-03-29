import numpy as np
import scipy.linalg as sla
import os
import sys
import shutil
import click
from glob import glob
from tqdm import tqdm, trange

import fym.logging as logging
from fym.agents.LQR import clqr
from fym.core import BaseEnv, BaseSystem, Sequential


class Env(BaseEnv):
    A = np.array([
        [0, 1, 0],
        [0, 0, 0],
        [1, 0, 0]])
    B = np.array([
        [0], [1], [0]])
    Wd = np.array([
        [-18.59521], [15.162375], [-62.45153], [9.54708], [21.45291]])
    Qm = np.diag([2800, 1, 15000])
    Rm = np.diag([50])
    Q = np.eye(3)
    K, *_ = clqr(A, B, Qm, Rm)
    Am = A - B.dot(K)
    Bm = np.array([[0], [0], [-1]])
    P = sla.solve_lyapunov(Am.T, -Q)

    def __init__(self):
        super().__init__(dt=1e-2, max_t=40)
        self.base = Sequential(
            main=BaseSystem(np.array([[0.3, 0, 0]]).T),
            model=BaseSystem(np.array([[0.3, 0, 0]]).T),
            command=SineWaveGenerator(15, 0),
        )

        self.logger_callback = self.get_logger_callback
        self.eager_stop = self.get_eager_stop

    def set_ctrl(self, ctrl):
        self.ctrl = ctrl
        self.expdir = "-".join([self.envdir, ctrl.name])

    def set_envdir(self, casedir):
        self.envdir = os.path.join(casedir, self.name)

    def set_dot(self, t):
        x, xm, cstate = self.base.observe_list()
        c = cstate[0:1]

        u = self.ctrl.get(self.observe_dict())

        self.base.main.dot = (
            self.A.dot(x)
            + self.B.dot(u + self.get_delta(x))
            + self.Bm.dot(c)
        )
        self.base.model.dot = self.Am.dot(x) + self.Bm.dot(c)
        self.base.command.set_dot(cstate)
        self.ctrl.set_dot(x, xm, cstate, u)

    def step(self):
        *_, done = self.update()
        done = done or self.clock.time_over()
        return done

    def get_logger_callback(self, i, t, y, *args):
        states = self.observe_dict(y)
        x = states["base"]["main"]
        xm = states["base"]["model"]
        cstate = states["base"]["command"]
        u = self.ctrl.get(states)
        return {
            "time": t,
            "state": x,
            "model": xm,
            "command": cstate[0],
            "control": u
        }

    def get_eager_stop(self, thist, odehist):
        index = np.where(
            np.any(np.abs(odehist[:, (0, 1)]) > 5, axis=1))[0]
        if index.size == 0:
            done = False
        else:
            thist = thist[:index[0] + 1]
            odehist = odehist[:index[0] + 1]
            done = True
        return thist, odehist, done

    def get_delta(self, x):
        return self.Wd.T.dot(self.phid(x))

    def phid(self, x):
        return np.vstack((x[:2], np.abs(x[:2]) * x[1], x[0]**3))


class BaseCtrl:
    def __init__(self, K):
        super().__init__()
        self.K = K
        self.name = "BaseCtrl"

    def get(self, states):
        x = states["base"]["main"]
        return -self.K.dot(x)

    def set_dot(self, *args):
        pass


class Mrac(BaseCtrl, BaseEnv):
    def __init__(self, env):
        super().__init__(env.K)
        self.param = BaseSystem(np.zeros_like(env.Wd))
        self.gamma = 1e3
        self.P = env.P
        self.B = env.B
        self.phid = env.phid
        self.name = "MRAC"

    def get(self, states):
        base = super().get(states)
        x = states["base"]["main"]
        W = states["ctrl"]["param"]
        return base - W.T.dot(self.phid(x))

    def set_dot(self, x, xm, *args):
        e = x - xm
        self.param.dot = (
            self.gamma * self.phid(x).dot(e.T).dot(self.P).dot(self.B))


class NewMrac(Mrac):
    def __init__(self, env):
        super().__init__(env)
        self.actor = BaseSystem()
        self.critic = BaseSystem()

        tau = 1e-2
        self.filter = Sequential(
            phic=FilterSystem(tau),
            q=FilterSystem(tau),
            xrx=FilterSystem(tau),
            xrp=FilterSystem(tau),
            prp=FilterSystem(tau),
            pru=FilterSystem(tau),
        )
        self.tau = tau
        self.name = "NewMRAC"

    def set_dot(self, x, xm, cstate, u):
        super().set_dot(x, xm)
        wa, wc = self.actor.state, self.critic.state
        phicf, qf, xrxf, xrpf, prpf, pruf = self.filter.observe_list()

        K, R = self.K, self.R
        q = self.get_q(x)
        phic = self.get_phic(x)
        phia = self.get_phia(x)

        self.filter.phic.set_dot(phic)
        self.filter.q.set_dot(q)
        self.filter.xrx.set_dot(x.T.dot(K.T).dot(R).dot(K).dot(x))
        self.filter.xrp.set_dot(x.T.dot(K.T).dot(R).dot(phia.T))
        self.filter.prp.set_dot(phia.dot(R).dot(phia.T))
        self.filter.prp.set_dot(phia.dot(R).dot(u))

        error = (
            wc.T.dot(phi - phif) / self.tau
            + 2 * wa.T.dot(pruf - prp.dot(wa))
        self.critic.dot = 1 / self.tau * (phi - phif) * error


class FilterSystem(BaseSystem):
    def __init__(self, initial_state, tau):
        super().__init__(initial_state)
        self.tau = tau

    def set_dot(self, signal):
        self.dot = -1 / self.tau * (self.state - signal)


class SineWaveGenerator(BaseSystem):
    """ y(t) = sin(2 * pi / period * t + phase) """
    def __init__(self, period, phase):
        super().__init__(np.array([[np.sin(phase)], [np.cos(phase)]]))
        w = 2 * np.pi / period
        self.A = np.array([[0, 1], [-w, 0]])

    def set_dot(self, c):
        self.dot = self.A.dot(c)


def _run(env, num, use_tqdm=False):
    expdir = env.expdir
    custom_range = trange if tqdm else range
    for i in custom_range(num):
        path = "-".join([expdir, f"{i:03d}.h5"])
        env.logger = logging.Logger(path=path)
        env.reset()
        while True:
            done = env.step()
            if done:
                break

        env.close()


def _case1():
    # Case 1 : Different exploration data (different behavior)
    print("Case 1")
    print("======")
    casedir = os.path.join("data", "case1")
    env = Env()
    env.set_envdir(casedir)

    ctrls = [BaseCtrl(env.K), Mrac(env)]
    # ctrls = [Mrac(env)]
    for ctrl in ctrls:
        env.set_ctrl(ctrl)
        _run(env, 1)


@click.group()
def main():
    pass


@main.command()
@click.option("--case", "-c", multiple=True, type=int)
@click.option("--pass-yes", "-y", is_flag=True)
def run(case, pass_yes):
    np.random.seed(0)
    caselist = {1: _case1}

    if not case:
        if os.path.exists("data"):
            if pass_yes or input(
                    f"Delete \"data\"? [Y/n]: ") in ["", "Y", "y"]:
                shutil.rmtree("data")
            else:
                sys.exit()

        for case_run in caselist.values():
            case_run()
    else:
        dupdirs = []
        for c in case:
            casedir = os.path.join("data", "case" + str(c))
            if os.path.exists(casedir):
                dupdirs.append(casedir)

        if dupdirs:
            if pass_yes or input(
                    f"Delete {', '.join(dupdirs)}? [Y/n]: ") in ["", "Y", "y"]:
                for d in dupdirs:
                    shutil.rmtree(d)
            else:
                sys.exit()

        for c in case:
            caselist[c]()


def draw_true(axes):
    # Draw true parameters
    for i in (1, 0.5, 0):
        axes[0].axhline(i, c="r", ls="--")

    for i in (-1, -2):
        axes[1].axhline(i, c="r", ls="--")


def set_axes(fig, axes):
    fig.tight_layout()
    fig.subplots_adjust(top=0.9, left=0.1)


def _plot_case1():
    import matplotlib.pyplot as plt
    import os
    from glob import glob
    from cycler import cycler

    import fym.logging as logging

    plt.rc("font", **{
        "family": "sans-serif",
        "sans-serif": ["Helvetica"],
    })
    plt.rc("text", usetex=True)
    plt.rc("lines", linewidth=1)
    plt.rc("axes", grid=True)
    plt.rc("axes", prop_cycle=cycler(color="k"))
    plt.rc("grid", linestyle="--", alpha=0.8)
    plt.rc("figure", figsize=[6, 4])

    casedir = os.path.join("data", "case1")
    samplelist = sorted(glob(os.path.join(casedir, "*.h5")))

    custom_cycler = (
        cycler(ls=["-", "--", "-.", ":"])
    )

    fig, axes = plt.subplots(3, 1, sharex=True)
    for samplepath, cc in zip(samplelist, custom_cycler()):
        data = logging.load(samplepath)
        t = data["time"]
        x = data["state"].squeeze()
        xm = data["model"].squeeze()
        u = data["control"].squeeze()
        axes[0].plot(t, x[:, 0], c="k", ls="-")
        axes[0].plot(t, xm[:, 0], c="r", ls="--")
        axes[1].plot(t, x[:, 1], c="k", ls="-")
        axes[1].plot(t, xm[:, 1], c="r", ls="--")
        axes[2].plot(t, u, **cc)

    axes[0].set_ylabel(r"$x_1$")
    axes[1].set_ylabel(r"$x_1$")
    axes[2].set_ylabel(r"$u$")
    axes[2].set_xlabel(r"Time [s]")
    set_axes(fig, axes)
    plt.show()


@main.command()
@click.option("--case", "-c", multiple=True, type=int)
def plot(case):
    import matplotlib.pyplot as plt
    import os
    from glob import glob
    from cycler import cycler

    import fym.logging as logging

    plt.rc("font", **{
        "family": "sans-serif",
        "sans-serif": ["Helvetica"],
    })
    plt.rc("text", usetex=True)
    plt.rc("lines", linewidth=1)
    plt.rc("axes", grid=True)
    plt.rc("axes", prop_cycle=cycler(color="k"))
    plt.rc("grid", linestyle="--", alpha=0.8)
    plt.rc("figure", figsize=[6, 4])

    caselist = {1: _plot_case1}

    if not case:
        for case_run in caselist.values():
            case_run()
    else:
        for c in case:
            caselist[c]()

    plt.show()


if __name__ == "__main__":
    main()
