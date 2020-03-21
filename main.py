import shutil
import os
import sys
import numpy as np
import scipy.integrate as sint
import click
from glob import glob
from functools import partial
from tqdm import tqdm, trange

from torch.utils.data import DataLoader, Dataset

import fym.core as core
import fym.logging as logging


class Linear(core.BaseSystem):
    """ u = Phi(x)^T w """
    def __init__(self, w, phi):
        super().__init__(w)
        self.noiselist = []
        if phi:
            self.set_phi(phi)

    def set_phi(self, phi):
        self.phi = phi

    def add_noise(self, noise):
        self.noiselist.append(noise)

    def get_noise(self, t):
        res = 0
        if self.noiselist:
            res = np.sum([noise(t) for noise in self.noiselist], axis=0)
        return res

    def get(self, t, x, w=None):
        if w is None:
            w = self.state
        ut = self.phi(x).T.dot(w)
        ut += self.get_noise(t)
        return ut


def lewis_noise(t, noisescale):
    if t <= np.pi:
        return noisescale * (
            np.sin(t)**2 * np.cos(t)
            + np.sin(2 * t)**2 * np.cos(0.1 * t)
            + np.sin(-1.2 * t)**2 * np.cos(0.5 * t)
            + np.sin(t)**5
        )
    else:
        return 0


class Env(core.BaseEnv):
    def __init__(self, phia, paramvar=0.3, noisescale=0.8, k=20):
        super().__init__(dt=0.01, max_t=5)
        self.main = core.BaseSystem(np.zeros((2, 1)))
        self.inner_ctrl = Linear(-np.array([[1, 2]]).T, phi=phia)
        self.inner_ctrl.add_noise(partial(lewis_noise, noisescale=noisescale))
        self.filter = core.BaseSystem(np.zeros((2, 1)))

        self.paramvar = paramvar
        self.k = k
        self.name = "-".join(["env", f"{paramvar:.5f}", f"{noisescale:.5f}"])

        self.logger_callback = self.get_info

    def reset(self, mode):
        super().reset()
        if mode == "random":
            self.main.state = np.random.randn(2, 1)
            self.inner_ctrl.state = (
                self.inner_ctrl.state
                + self.paramvar * (np.random.rand() - 0.5)
            )
            self.filter.state = self.main.state

    def set_dot(self, t):
        x = self.main.state
        xf = self.filter.state
        u = self.inner_ctrl.get(t, x)
        self.main.dot = self.deriv(x, u)
        self.inner_ctrl.dot = np.zeros_like(self.inner_ctrl.state)
        self.filter.dot = -self.k * (xf - x)

    def deriv(self, x, u):
        x1, x2 = x
        x1dot = -x1 + x2
        x2dot = (
            -0.5 * x1 - 0.5 * x2 * (1 - (np.cos(2 * x1) + 2) ** 2)
            + (np.cos(2 * x1) + 2) * u
        )
        return np.vstack((x1dot, x2dot))

    def step(self):
        self.update()
        done = self.clock.time_over()
        return done

    def get_info(self, i, t, y, t_hist, ode_hist):
        x, w, xf = self.observe_list(y)
        u = self.inner_ctrl.get(t, x, w)
        return {
            "time": t,
            "state": x,
            "control": u,
            "fstate": xf,
            "fstate_dot": -self.k * (xf - x),
        }

    def set_envdir(self, casedir):
        self.envdir = os.path.join(casedir, self.name)


class BaseAgent(core.BaseEnv):
    Q = np.eye(2)
    R = np.ones((1, 1))

    def __init__(self):
        super().__init__()

    def get_info(self):
        raise NotImplementedError

    def set_datalist(self, datalist):
        self.datalist = datalist

    def set_trainpath(self, envdir):
        self.trainpath = os.path.join(
            envdir,
            "-".join(["train", self.name + ".h5"])
        )

    def policy_evaluation(self):
        philist = []
        ylist = []
        for data in self.datalist:
            phi, y = self.processing(data, self.wa.state)
            philist += phi
            ylist += y

        philist, ylist = np.stack(philist), np.stack(ylist)

        # Least square
        w = np.linalg.pinv(philist.squeeze()).dot(ylist)
        self.state = w.ravel()
        return self.observe_dict()

    def processing(self):
        raise NotImplementedError

    def cost(self, x, u):
        return x.T.dot(self.Q).dot(x) + u.T.dot(self.R).dot(u)

    def phic(self, x):
        return np.vstack((x ** 2, x[0] * x[1]))


class RADP(BaseAgent):
    def __init__(self, x_size, u_size, phia, T):
        super().__init__()
        self.phia = phia
        self.vphia = np.vectorize(self.phia, signature="(n,1)->(p,m)")
        self.vphic = np.vectorize(self.phic, signature="(n,1)->(p,1)")
        self.vcost = np.vectorize(self.cost, signature="(n,1),(m,1)->()")

        x = np.zeros((x_size, 1))
        # wa_init = np.zeros((self.phia(x).shape[0], u_size))
        wa_init = -np.array([[1, 2]]).T * 0.6
        self.wa = core.BaseSystem(wa_init)
        self.wc = core.BaseSystem(np.zeros((self.phic(x).shape[0], 1)))
        self.T = T

        self.name = "-".join([self.name, f"{T:.4f}"])

    def get_info(self):
        return dict(
            classname=self.__class__.__name__,
            name=self.name,
            T=self.T,
        )

    def processing(self, data, wa):
        t = data["time"]
        x = data["state"]
        u = data["control"]
        ui = self.vphia(x).transpose(0, 2, 1).dot(wa)
        delu = u - ui
        phi2 = 2 * np.einsum(
            "bpm,bmo->bpo",
            self.vphia(x).dot(self.R),
            delu).squeeze()
        y = -self.vcost(x, ui)

        philist = []
        ylist = []
        indexlist = self.get_indexlist(t, self.T)
        for index in indexlist:
            xi = x[index]
            phi = np.vstack((
                np.atleast_1d(sint.trapz(phi2[index], t[index], axis=0))[:, None],
                self.phic(xi[-1]) - self.phic(xi[0])
            ))
            yi = np.atleast_1d(sint.trapz(y[index], t[index]))
            philist.append(phi)
            ylist.append(yi)

        return philist, ylist

    def get_indexlist(self, tlist, T):
        indexlist = []
        pi, pt = 0, 0
        for i, t in enumerate(tlist):
            if t > pt + T:
                indexlist.append(slice(pi, i))
                pi, pt = i - 1, tlist[i - 1]

        if pi + 1 != len(tlist):
            indexlist.append(slice(pi, len(tlist)))

        return indexlist


class IFRADP(BaseAgent):
    def __init__(self, x_size, u_size, phia, env):
        super().__init__()
        self.phia = phia
        self.deriv = env.deriv
        self.vphic = np.vectorize(self.phic, signature="(n,1)->(p,1)")
        self.vphia = np.vectorize(self.phia, signature="(n,1)->(p,m)")
        self.vcost = np.vectorize(self.cost, signature="(n,1),(m,1)->()")
        self.vgrad_phic = np.vectorize(
            self.grad_phic, signature="(n,1)->(n,p)")
        self.vphic_dot = np.vectorize(
            self.phic_dot, signature="(n,1),(m,1)->(p,1)")

        x = np.zeros((x_size, 1))
        # wa_init = np.zeros((self.phia(x).shape[0], u_size))
        wa_init = -np.array([[1, 2]]).T * 0.6
        self.wa = core.BaseSystem(wa_init)
        self.wc = core.BaseSystem(np.zeros((self.phic(x).shape[0], 1)))

    def get_info(self):
        return dict(
            classname=self.__class__.__name__,
            name=self.name
        )

    def processing(self, data, wa):
        t = data["time"]
        x = data["state"]
        xf = data["fstate"]
        xf_dot = data["fstate_dot"]
        u = data["control"]
        ui = self.vphia(x).transpose(0, 2, 1).dot(wa)
        delu = u - ui
        grad_phic = self.vgrad_phic(xf)
        phi1 = np.einsum("bnp,bno->bpo", grad_phic, xf_dot)
        # phic = self.vphic(x).squeeze()
        # phi1 = np.vstack([np.gradient(p, t) for p in phic.T]).T[..., None]
        phi1_comp = self.vphic_dot(x, u)
        breakpoint()
        phi2 = 2 * np.einsum(
            "bpm,bmo->bpo",
            self.vphia(x).dot(self.R),
            delu)
        y = -self.vcost(x, ui)

        philist = []
        ylist = []
        for i in range(len(t)):
            phi = np.vstack((phi2[i], phi1[i]))
            philist.append(phi)
            ylist.append(y[i])

        return philist, ylist

    def grad_phic(self, x):
        return np.hstack((2 * np.diag(x.ravel()), np.flip(x)))

    def phic_dot(self, x, u):
        return self.grad_phic(x).T.dot(self.deriv(x, u))


def _sample(env, num, tqdm=False):
    envdir = env.envdir
    custom_range = trange if tqdm else range
    for i in custom_range(num):
        path = os.path.join(envdir, f"sample-{i:03d}.h5")
        env.logger = logging.Logger(path=path)
        env.reset("random")
        while True:
            done = env.step()
            if done:
                break

        env.close()


def _train(agent):
    trainpath = agent.trainpath
    logger = logging.Logger(path=trainpath)
    logger.set_info(agent.get_info())

    params = agent.observe_dict()
    eps = 1e-10
    for epoch in range(50):
        params_next = agent.policy_evaluation()

        logger.record(epoch=epoch, params=params)

        if np.linalg.norm(params_next["wc"] - params["wc"]) < eps:
            break

        params = params_next

    logger.close()


@click.group()
def main():
    pass


def _case1():
    # Case 1 : Different exploration data (different behavior)
    print("Case 1")
    print("======")
    casedir = os.path.join("data", "case1")

    def phia(x):
        x1, x2 = x
        return np.vstack((x2 * np.cos(2 * x1), x2))

    envargs = np.random.rand(10, 2)

    for paramvar, noisescale in tqdm(envargs):
        env = Env(phia, paramvar, noisescale)
        env.set_envdir(casedir)
        _sample(env, 20)

        # Train
        samplelist = sorted(glob(os.path.join(env.envdir, "sample-*.h5")))
        datalist = [logging.load(path) for path in samplelist]

        agent = RADP(2, 1, phia, 0.05)
        agent.set_datalist(datalist)
        agent.set_trainpath(env.envdir)
        _train(agent)


def _case2():
    # Case 2 - Different integral time step
    print("Case 2")
    print("======")
    casedir = os.path.join("data", "case2")

    def phia(x):
        x1, x2 = x
        return np.vstack((x2 * np.cos(2 * x1), x2))

    paramvar, noisescale = np.random.rand(2)
    env = Env(phia, paramvar, noisescale*0)
    env.set_envdir(casedir)
    _sample(env, 20, tqdm=True)

    samplelist = sorted(glob(os.path.join(env.envdir, "sample-*.h5")))
    datalist = [logging.load(path) for path in samplelist]

    Tlist = [0.05, 0.5, 2]
    agentlist = [IFRADP(2, 1, phia, env)]
    agentlist += [RADP(2, 1, phia, T) for T in Tlist]

    for agent in tqdm(agentlist):
        agent.set_datalist(datalist)
        agent.set_trainpath(env.envdir)
        _train(agent)


def _case3():
    # Case 3 - Inaccurate basis function
    print("Case 3")
    print("======")
    casedir = os.path.join("data", "case3")

    def phia(x):
        return x

    paramvar, noisescale = np.random.rand(2)
    env = Env(phia, paramvar, noisescale*0)
    env.set_envdir(casedir)
    _sample(env, 20, tqdm=True)

    samplelist = sorted(glob(os.path.join(env.envdir, "sample-*.h5")))
    datalist = [logging.load(path) for path in samplelist]

    Tlist = [0.05, 0.5, 2]
    agentlist = [RADP(2, 1, phia, T) for T in Tlist]
    agentlist += [IFRADP(2, 1, phia, env)]

    for agent in tqdm(agentlist):
        agent.set_datalist(datalist)
        agent.set_trainpath(env.envdir)
        _train(agent)


@main.command()
@click.option("--case", "-c", multiple=True, type=int)
def train(case):
    np.random.seed(0)
    caselist = {1: _case1, 2: _case2, 3: _case3}

    if not case:
        if os.path.exists("data"):
            if input(f"Delete \"data\"? [Y/n]: ") in ["", "Y", "y"]:
                shutil.rmtree("data")
            else:
                sys.exit()

        for case_run in caselist.values():
            case_run()
    else:
        if input(f"Delete case{case}? [Y/n]: ") in ["", "Y", "y"]:
            for c in case:
                casedir = os.path.join("data", "case" + str(c))
                shutil.rmtree(casedir)
        else:
            sys.exit()

        for c in case:
            caselist[c]()


@main.command()
def plot():
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

    def draw_true(axes):
        # Draw true parameters
        for i in (1, 0.5, 0):
            axes[0].axhline(i, c="r", ls="--")

        for i in (-1, -2):
            axes[1].axhline(i, c="r", ls="--")

    def set_axes(fig, axes):
        axes[0].set_ylabel(r"$w_c$")
        axes[0].set_xlim(left=0)
        axes[1].set_ylabel(r"$w_a$")
        axes[1].set_xlabel("Iteration")

        fig.tight_layout()
        fig.subplots_adjust(top=0.9, left=0.1)

    # Case 1
    casedir = os.path.join("data", "case1")
    trainlist = glob(os.path.join(casedir, "*", "train-*.h5"))

    fig, axes = plt.subplots(2, 1, sharex=True)
    draw_true(axes)

    for trainpath in trainlist:
        data = logging.load(trainpath)
        epoch = data["epoch"]
        wa = data["params"]["wa"].squeeze()
        wc = data["params"]["wc"].squeeze()

        axes[0].plot(epoch, wc)
        axes[1].plot(epoch, wa)

        axes[0].set_ylabel(r"$w_c$")
        axes[0].set_xlim(left=0)
        axes[1].set_ylabel(r"$w_a$")
        axes[1].set_xlabel("Iteration")

    set_axes(fig, axes)

    # Case 2
    casedir = os.path.join("data", "case2")
    trainlist = sorted(glob(os.path.join(casedir, "*", "train-*.h5")))

    custom_cycler = (
        cycler(color=["k"])
        * cycler(ls=["-", "--", "-.", ":"])
    )

    fig, axes = plt.subplots(2, 1, sharex=True)
    draw_true(axes)

    legendlines = []
    for trainpath, cc in zip(trainlist, custom_cycler):
        data, info = logging.load(trainpath, with_info=True)
        if info["classname"] == "RADP":
            label = rf"RADP ($T = {info['T']}$)"
        else:
            label = r"IFRADP"
            cc = dict(color="b", ls="-", zorder=10)
        epoch = data["epoch"]
        wa = data["params"]["wa"].squeeze()
        wc = data["params"]["wc"].squeeze()

        lines = axes[0].plot(epoch, wc, **cc, label=label)
        legendlines.append(lines[0])
        axes[1].plot(epoch, wa, **cc)

    fig.legend(
        handles=legendlines,
        bbox_to_anchor=(0.1, 0.92, 0.87, .05),
        loc='lower center',
        ncol=4,
        mode="expand",
        borderaxespad=0.
    )
    set_axes(fig, axes)

    fig, axes = plt.subplots(2, 1)

    legendlines = []
    for trainpath, cc in zip(trainlist, custom_cycler):
        data, info = logging.load(trainpath, with_info=True)
        if info["classname"] == "RADP":
            label = rf"RADP ($T = {info['T']}$)"
        else:
            label = r"IFRADP"
            cc = dict(color="b", ls="-", zorder=10)
        epoch = data["epoch"]
        wa = data["params"]["wa"].squeeze()
        ewa = np.linalg.norm(wa - [-1, -2], axis=1)
        wc = data["params"]["wc"].squeeze()
        ewc = np.linalg.norm(wc - [0.5, 1, 0], axis=1)

        lines = axes[0].plot(epoch, ewc, **cc, label=label)
        legendlines.append(lines[0])
        axes[1].plot(epoch, ewa, **cc)

        axes[0].set_yscale("log")
        axes[1].set_yscale("log")

    fig.legend(
        handles=legendlines,
        bbox_to_anchor=(0.1, 0.92, 0.87, .05),
        loc='lower center',
        ncol=4,
        mode="expand",
        borderaxespad=0.
    )
    set_axes(fig, axes)

    # Case 3
    casedir = os.path.join("data", "case3")
    trainlist = sorted(glob(os.path.join(casedir, "*", "train-*.h5")))

    custom_cycler = (
        cycler(color=["k"])
        * cycler(ls=["-", "--", "-.", ":"])
    )

    def draw_true(axes):
        # Draw true parameters
        for i in (1, 0.5, 0):
            axes[0].axhline(i, c="r", ls="--")

        for i in (0, -3):
            axes[1].axhline(i, c="r", ls="--")

    fig, axes = plt.subplots(2, 1, sharex=True)
    draw_true(axes)

    legendlines = []
    for trainpath, cc in zip(trainlist, custom_cycler):
        data, info = logging.load(trainpath, with_info=True)
        if info["classname"] == "RADP":
            label = rf"RADP ($T = {info['T']}$)"
        else:
            label = r"IFRADP"
            cc = dict(color="b", ls="-", zorder=10)
        epoch = data["epoch"]
        wa = data["params"]["wa"].squeeze()
        wc = data["params"]["wc"].squeeze()

        lines = axes[0].plot(epoch, wc, **cc, label=label)
        legendlines.append(lines[0])
        axes[1].plot(epoch, wa, **cc)

    fig.legend(
        handles=legendlines,
        bbox_to_anchor=(0.1, 0.92, 0.87, .05),
        loc='lower center',
        ncol=4,
        mode="expand",
        borderaxespad=0.
    )
    set_axes(fig, axes)

    plt.show()


if __name__ == "__main__":
    """
    Sample code:
        ```bash
        $ python main.py train
        $ python main.py plot
        ```
    """
    main()
