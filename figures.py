import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

import fym.logging as logging

plt.rc("font", **{
    "family": "sans-serif",
    "sans-serif": ["Helvetica"],
})
plt.rc("text", usetex=True)
plt.rc("lines", linewidth=1)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=0.8)
plt.rc("figure", figsize=[6, 4])


def plot_single(path, canvas=[], name=None, **kwargs):
    data = logging.load(path)

    def create_canvas():
        fig, axes = plt.subplots(2, 2, sharex=True, num="states")
        for ax in axes.flat:
            ax.mod = 1
        axes[0, 0].set_ylabel(r"$V_T$ [m/s]")
        axes[0, 1].set_ylabel(r"$\alpha$ [deg]")
        axes[0, 1].mod = np.rad2deg(1)
        axes[1, 0].set_ylabel(r"$q$ [deg/s]")
        axes[1, 0].mod = np.rad2deg(1)
        axes[1, 1].set_ylabel(r"$\gamma$ [deg]")
        axes[1, 1].mod = np.rad2deg(1)
        axes[1, 0].set_xlabel("time [s]")
        axes[1, 1].set_xlabel("time [s]")
        canvas.append((fig, axes))

        fig, axes = plt.subplots(2, 2, sharex=True, num="control")
        for ax in axes.flat:
            ax.mod = 1
        axes[0, 0].set_ylabel(r"$\delta_t$")
        axes[0, 1].set_ylabel(r"$\delta_e$ [deg]")
        axes[0, 1].mod = np.rad2deg(1)
        axes[1, 0].set_ylabel(r"$\eta_1$")
        axes[1, 1].set_ylabel(r"$\eta_2$")
        axes[1, 0].set_xlabel("time [s]")
        axes[1, 1].set_xlabel("time [s]")
        canvas.append((fig, axes))

        return canvas

    if not canvas:
        canvas = create_canvas()

    time = data["time"]

    legend_line = []

    fig, axes = canvas[0]
    for ax, x in zip(axes.flat, data["state"].T):
        ln, = ax.plot(time, x * ax.mod, **kwargs)
    fig.tight_layout()
    legend_line.append(ln)

    fig, axes = canvas[1]
    for ax, u in zip(axes.flat, data["action"].T):
        ln, = ax.plot(time, u * ax.mod, **kwargs)
    fig.tight_layout()
    legend_line.append(ln)

    if name is not None:
        for ln in legend_line:
            ln.set_label(name)

        for window in canvas:
            fig, axes = window
            axes[0, 0].legend(*axes[-1, -1].get_legend_handles_labels())

    return canvas


def plot_mult(dataset, color_cycle=None, names=None):
    canvas = None
    if color_cycle is None:
        color_cycle = cycler(
            color=plt.rcParams["axes.prop_cycle"].by_key()["color"]
        )

    if names is not None:
        for data, color, name in zip(dataset.values(), color_cycle(), names):
            plot_single(data, color=color["color"], name=name)

        for fig, axes in canvas:
            axes[0].legend(*axes[0].get_legend_handles_labels())
    else:
        for (name, data), color in zip(dataset.items(), color_cycle()):
            plot_single(data, color=color["color"], name=name)

    plt.show()


def train_plot(savepath, **kwargs):
    data, info = logging.load(savepath, with_info=True)

    canvas = []
    fig, axes = plt.subplots(2, 2, num="hist")
    axes[0, 0].set_ylabel("Critic loss")
    axes[0, 1].set_ylabel("Actor loss")
    axes[1, 0].set_ylabel("Eigenvalues (real)")
    axes[1, 1].set_ylabel("Parameters")
    axes[0, 0].set_xlabel("Step")
    axes[0, 1].set_xlabel("Epoch")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 1].set_xlabel("Epoch")
    canvas.append((fig, axes))

    fig, axes = canvas[0]

    global_step = data["global_step"]

    RUNNING_KWARGS = dict(kwargs, ls="-", c="r")
    loss_critic = data["info"]["loss_critic"]
    rloss_critic = data["info"]["rloss_critic"]
    axes[0, 0].plot(global_step, loss_critic, **kwargs)
    axes[0, 0].plot(global_step, rloss_critic, **RUNNING_KWARGS)
    axes[0, 0].set_yscale("log")

    epoch = data["epoch"]

    loss_actor = data["loss_actor"]
    axes[0, 1].plot(epoch, loss_actor, **kwargs)

    TRUE_KWARGS = dict(kwargs, ls="--", c="k")
    eigvals = data["eigvals"]
    true_eigvals = np.tile(info["true_eigvals"].real, (len(epoch), 1))
    axes[1, 0].plot(epoch, eigvals, **kwargs)
    axes[1, 0].plot(epoch, true_eigvals, **TRUE_KWARGS)

    param = data["param"]
    true_param = np.tile(info["true_param"].ravel(), (len(epoch), 1))
    param = param.reshape(-1, param[0].size)
    axes[1, 1].plot(epoch, param, **kwargs)
    axes[1, 1].plot(epoch, true_param, **TRUE_KWARGS)

    fig.tight_layout()


def plot_gan(path):
    data = logging.load(path)

    xlabels = (
        r"$V_T$ [m/s]", r"$\alpha$ [deg]", r"$q$ [deg/s]", r"$\theta$ [deg]"
    )
    xmod = [1, np.rad2deg(1), np.rad2deg(1), np.rad2deg(1)]
    ulabels = (
        r"$\delta_t$", r"$\delta_e$", r"$\eta_1$", r"$\eta_2$"
    )
    umod = [1, np.rad2deg(1), 1, 1]

    def plot_xu(canvas, i, j, data):
        x, u, fake_u = (data[k] for k in ["state", "action", "fake_action"])

        x = x[:, j] * xmod[j]
        u = u[:, i] * umod[i]
        fake_u = fake_u[:, i] * xmod[i]

        xmin, xmax = x.min(), x.max()
        umin, umax = u.min(), u.max()

        fig, axes = canvas[0]

        axes[i, 0].set_ylabel(ulabels[i])
        axes[-1, 2 * j].set_xlabel(xlabels[j])
        axes[-1, 2 * j + 1].set_xlabel(xlabels[j])

        axes[i, 2 * j].set_xlim([xmin, xmax])
        axes[i, 2 * j + 1].set_xlim([xmin, xmax])
        axes[i, 2 * j].set_ylim([umin, umax])

        axes[0, 2 * j].set_title("Real")
        axes[0, 2 * j + 1].set_title("Fake")

        ax = axes[i, 2 * j]
        ax.plot(x, u, markersize=2, mew=0, mfc=(0, 0, 0, 1))

        ax = axes[i, 2 * j + 1]
        ax.plot(x, fake_u, markersize=2, mew=0, mfc=(0, 0, 0, 1))

    canvas = []

    fig, axes = plt.subplots(
        4, 8, sharex=True, sharey=True, squeeze=False, num="real"
    )
    canvas.append((fig, axes))

    for i in range(4):
        for j in range(4):
            plot_xu(canvas, i, j, data)

    fig.tight_layout()


def plot_hist(path):
    # path = os.path.join(
    #     obj.gan_dir,
    #     os.path.relpath(os.path.dirname(testfile), obj.test_dir),
    #     "train_history.h5"
    # )

    histdata = logging.load(path)

    canvas = []

    fig, axes = plt.subplots(1, 2, sharey=True, squeeze=False, num="loss")
    axes[0, 0].set_ylabel(r"Loss")

    axes[0, 0].set_xlabel(r"Epoch")
    axes[0, 1].set_xlabel(r"Epoch")

    axes[0, 0].set_title("Generator")
    axes[0, 1].set_title("Discrimator")

    canvas.append((fig, axes))

    fig, axes = canvas[0]
    ax = axes[0, 0]
    ax.plot(histdata["epoch"], histdata["loss_g"])
    ax = axes[0, 1]
    ax.plot(histdata["epoch"], histdata["loss_d"])
    fig.tight_layout()


def show():
    plt.show()
