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
        ut = np.atleast_2d(self.phi(x).T.dot(w))
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
    Q = np.eye(2)
    R = np.ones((1, 1))

    def __init__(self, paramvar=0.3, noisescale=0.8, k=1e2,
                 phia=None, phic=None):
        super().__init__(dt=1e-2, max_t=5)

        self.phia = phia or self.phia
        self.phic = phic or self.phic

        self.main = core.BaseSystem(np.zeros((2, 1)))
        self.inner_ctrl = Linear(-np.array([[1, 2]]).T, phi=self.phia)
        self.inner_ctrl.add_noise(partial(lewis_noise, noisescale=noisescale))

        x = np.zeros((2, 1))
        u = np.zeros((1, 1))

        phic, q, prp, pru = self.get_filter_inputs(x, u)
        self.filter = core.Sequential(
            phic=core.BaseSystem(phic),
            q=core.BaseSystem(q),
            prp=core.BaseSystem(prp),
            pru=core.BaseSystem(pru)
        )

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
            x = self.main.state
            self.filter.phic.state = self.phic(x)
            for system in (self.filter.q, self.filter.prp, self.filter.pru):
                system.state = np.zeros_like(system.state)

    def get_filter_inputs(self, x, u):
        return [self.phic(x),
                self.get_q(x),
                self.get_prp(x),
                self.get_pru(x, u)]

    def set_dot(self, t):
        x = self.main.state
        u = self.inner_ctrl.get(t, x)
        self.main.dot = self.deriv(x, u)
        self.inner_ctrl.dot = np.zeros_like(self.inner_ctrl.state)
        self.set_filter_dot(x, u)

    def deriv(self, x, u):
        x1, x2 = x
        x1dot = -x1 + x2
        x2dot = (
            -0.5 * x1 - 0.5 * x2 * (1 - (np.cos(2 * x1) + 2) ** 2)
            + (np.cos(2 * x1) + 2) * u
        )
        return np.vstack((x1dot, x2dot))

    def set_filter_dot(self, x, u):
        fis = self.get_filter_inputs(x, u)
        for system, fi in zip(self.filter.systems, fis):
            system.dot = -self.k * (system.state - fi)

    def step(self):
        self.update()
        done = self.clock.time_over()
        return done

    def get_info(self, i, t, y, t_hist, ode_hist):
        states = self.observe_dict(y)
        x = states["main"]
        w = states["inner_ctrl"]
        u = self.inner_ctrl.get(t, x, w)
        return {
            "time": t,
            "state": x,
            "control": u,
            "filter": states["filter"]
        }

    def set_envdir(self, casedir):
        self.envdir = os.path.join(casedir, self.name)

    def get_prp(self, x):
        phia = self.phia(x)
        return phia.dot(self.R).dot(phia.T)

    def get_pru(self, x, u):
        return self.phia(x).dot(self.R).dot(u)

    def get_q(self, x):
        return x.T.dot(self.Q).dot(x)

    def phia(self, x):
        x1, x2 = x
        return np.vstack((x2 * np.cos(2 * x1), x2))

    def phic(self, x):
        return np.vstack((x ** 2, x[0] * x[1]))

    def get_cost(self, x, u):
        return self.get_q(x) + u.T.dot(self.R).dot(u)


class BaseAgent(core.BaseEnv):
    def __init__(self, env):
        super().__init__()
        self.phia = env.phia
        self.phic = env.phic
        self.Q = env.Q
        self.R = env.R
        self.get_cost = env.get_cost

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
        w = np.linalg.pinv(philist.squeeze()).dot(ylist.squeeze())
        self.state = w.ravel()
        return self.observe_dict()

    def processing(self):
        raise NotImplementedError


class RADP(BaseAgent):
    def __init__(self, x_size, u_size, T, env, phia=None, phic=None):
        super().__init__(env)
        self.phia = phia or self.phia
        self.phic = phic or self.phic
        self.vphia = np.vectorize(self.phia, signature="(n,1)->(p,m)")
        self.vphic = np.vectorize(self.phic, signature="(n,1)->(p,1)")
        self.vcost = np.vectorize(self.get_cost, signature="(n,1),(m,1)->()")

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
        phia = 2 * np.einsum(
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
                np.atleast_1d(sint.trapz(phia[index], t[index], axis=0))[:, None],
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
    """Integral-Free RADP"""
    def __init__(self, x_size, u_size, env, phia=None, phic=None):
        super().__init__(env)
        self.phia = phia or self.phia
        self.phic = phic or self.phic
        self.k = env.k
        self.deriv = env.deriv
        self.vphia = np.vectorize(self.phia, signature="(n,1)->(p,m)")
        self.vphic = np.vectorize(self.phic, signature="(n,1)->(p,1)")
        self.vcost = np.vectorize(self.get_cost, signature="(n,1),(m,1)->()")

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
        n = len(data["time"])
        # dn = int(n / 100)
        # cutindex = slice(10, int(n), dn)
        cutindex = slice(n)
        t = data["time"][cutindex]
        x = data["state"][cutindex, ...]
        phicf, qf, prpf, pruf = (
            data["filter"][k][cutindex, ...]
            for k in ("phic", "q", "prp", "pru"))

        phia = 2 * (pruf - prpf.dot(wa))
        phic = self.k * (self.vphic(x) - phicf)
        y = -qf - np.einsum("po,bpq,qo->b", wa, prpf, wa)[:, None, None]

        philist = []
        ylist = []
        for i in range(len(t)):
            phi = np.vstack((phia[i], phic[i]))
            philist.append(phi)
            ylist.append(y[i])

        return philist, ylist


class NDRADP(IFRADP):
    """Numerical Differentiator RADP"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def processing(self, data, wa):
        t = data["time"]
        x = data["state"]
        u = data["control"]
        vphia = self.vphia(x)
        vphic = self.vphic(x)
        ui = vphia.transpose(0, 2, 1).dot(wa)
        phia = 2 * np.einsum("bpm,bmo->bpo", vphia.dot(self.R), u - ui)
        phic = np.vstack([
            np.gradient(v, t)
            for v in vphic.squeeze().T
        ]).T[..., None]
        y = -self.vcost(x, ui)

        philist = []
        ylist = []
        for i in range(len(t)):
            phi = np.vstack((phia[i], phic[i]))
            philist.append(phi)
            ylist.append(y[i])

        return philist, ylist


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
    eps = 1e-15
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


def _case1(skip_sample):
    # Case 1 : Different exploration data (different behavior)
    print("Case 1")
    print("======")
    casedir = os.path.join("data", "case1")

    envargs = np.random.rand(10, 2)

    for paramvar, noisescale in tqdm(envargs):
        env = Env(paramvar, noisescale)
        env.set_envdir(casedir)
        if not skip_sample:
            _sample(env, 20)

        # Train
        samplelist = sorted(glob(os.path.join(env.envdir, "sample-*.h5")))
        datalist = [logging.load(path) for path in samplelist]

        agent = RADP(2, 1, 0.05, env)
        agent.set_datalist(datalist)
        agent.set_trainpath(env.envdir)
        _train(agent)


def _case2(skip_sample):
    # Case 2 - Different integral time step
    print("Case 2")
    print("======")
    casedir = os.path.join("data", "case2")

    paramvar, noisescale = np.random.rand(2)
    env = Env(paramvar, noisescale*0)
    env.set_envdir(casedir)
    if not skip_sample:
        _sample(env, 10, tqdm=True)

    samplelist = sorted(glob(os.path.join(env.envdir, "sample-*.h5")))
    datalist = [logging.load(path) for path in samplelist]

    Tlist = [0.02, 0.5, 2]
    agentlist = [NDRADP(2, 1, env)]
    agentlist += [IFRADP(2, 1, env)]
    agentlist += [RADP(2, 1, T, env) for T in Tlist]

    for agent in tqdm(agentlist):
        agent.set_datalist(datalist)
        agent.set_trainpath(env.envdir)
        _train(agent)


def _case3(skip_sample):
    # Case 3 - Inaccurate basis function
    print("Case 3")
    print("======")
    casedir = os.path.join("data", "case3")

    def phia(x):
        return x

    paramvar, noisescale = np.random.rand(2)
    env = Env(paramvar, noisescale*0, phia=phia)
    env.set_envdir(casedir)
    if not skip_sample:
        _sample(env, 10, tqdm=True)

    samplelist = sorted(glob(os.path.join(env.envdir, "sample-*.h5")))
    datalist = [logging.load(path) for path in samplelist]

    Tlist = [0.02, 0.5, 2]
    agentlist = [IFRADP(2, 1, env, phia=phia)]
    agentlist += [RADP(2, 1, T, env, phia=phia) for T in Tlist]

    for agent in tqdm(agentlist):
        agent.set_datalist(datalist)
        agent.set_trainpath(env.envdir)
        _train(agent)


@main.command()
@click.option("--case", "-c", multiple=True, type=int)
@click.option("--pass-yes", "-y", is_flag=True)
@click.option("--skip-sample", "-s", is_flag=True)
def train(case, pass_yes, skip_sample):
    np.random.seed(0)
    caselist = {1: _case1, 2: _case2, 3: _case3}

    if not case:
        if not skip_sample and os.path.exists("data"):
            if pass_yes or input(
                    f"Delete \"data\"? [Y/n]: ") in ["", "Y", "y"]:
                shutil.rmtree("data")
            else:
                sys.exit()

        for case_run in caselist.values():
            case_run(skip_sample)
    else:
        if not skip_sample:
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
            caselist[c](skip_sample)


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


def _plot_case2():
    import matplotlib.pyplot as plt
    from cycler import cycler

    # Case 2
    casedir = os.path.join("data", "case2")
    samplelist = sorted(glob(os.path.join(casedir, "*", "sample-*.h5")))
    trainlist = sorted(glob(os.path.join(casedir, "*", "train-*.h5")))

    custom_cycler = (
        cycler(color=["k"])
        * cycler(ls=["-", "--", "-.", ":"])
    )

    # plot samples
    fig, axes = plt.subplots(3, 1, sharex=True)
    for samplepath, cc in zip(samplelist, custom_cycler()):
        data = logging.load(samplepath)
        t = data["time"]
        x = data["state"].squeeze()
        u = data["control"].squeeze()
        axes[0].plot(t, x[:, 0], **cc)
        axes[1].plot(t, x[:, 1], **cc)
        axes[2].plot(t, u, **cc)

    axes[0].set_ylabel(r"$x_1$")
    axes[1].set_ylabel(r"$x_1$")
    axes[2].set_ylabel(r"$u$")
    axes[2].set_xlabel(r"Time [s]")
    set_axes(fig, axes)

    fig, axes = plt.subplots(2, 1, sharex=True)
    draw_true(axes)

    legendlines = []
    for trainpath, cc in zip(trainlist, custom_cycler()):
        data, info = logging.load(trainpath, with_info=True)
        if info["classname"] == "RADP":
            label = rf"RADP ($T = {info['T']}$)"
        elif info["classname"] == "IFRADP":
            label = r"IFRADP"
            cc = dict(color="b", ls="-", zorder=10)
        elif info["classname"] == "NDRADP":
            label = r"NDRADP"
            cc = dict(color="g", ls="-", zorder=10)
        epoch = data["epoch"]
        wa = data["params"]["wa"].squeeze()
        wc = data["params"]["wc"].squeeze()

        lines = axes[0].plot(epoch, wc, **cc, label=label)
        legendlines.append(lines[0])
        axes[1].plot(epoch, wa, **cc)

    axes[0].set_ylabel(r"$w_c$")
    axes[0].set_xlim(left=0)
    axes[1].set_ylabel(r"$w_a$")
    axes[1].set_xlabel("Iteration")

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
    for trainpath, cc in zip(trainlist, custom_cycler()):
        data, info = logging.load(trainpath, with_info=True)
        if info["classname"] == "RADP":
            label = rf"RADP ($T = {info['T']}$)"
        elif info["classname"] == "IFRADP":
            label = r"IFRADP"
            cc = dict(color="b", ls="-", zorder=10)
        elif info["classname"] == "NDRADP":
            label = r"NDRADP"
            cc = dict(color="g", ls="-", zorder=10)
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


def _plot_case3():
    import matplotlib.pyplot as plt
    from cycler import cycler

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

    axes[0].set_ylabel(r"$w_c$")
    axes[0].set_xlim(left=0)
    axes[1].set_ylabel(r"$w_a$")
    axes[1].set_xlabel("Iteration")

    fig.legend(
        handles=legendlines,
        bbox_to_anchor=(0.1, 0.92, 0.87, .05),
        loc='lower center',
        ncol=4,
        mode="expand",
        borderaxespad=0.
    )
    set_axes(fig, axes)


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

    caselist = {1: _plot_case1, 2: _plot_case2, 3: _plot_case3}

    if not case:
        for case_run in caselist.values():
            case_run()
    else:
        for c in case:
            caselist[c]()

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
