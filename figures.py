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
    data = logging.load(savepath)

    canvas = []
    fig, axes = plt.subplots(2, 2, sharex=True, num="hist")
    axes[0, 0].set_ylabel(r"$\delta$")
    axes[0, 1].set_ylabel(r"G Loss")
    axes[1, 0].set_ylabel(r"$w, v$")
    axes[1, 1].set_ylabel(r"$\theta$")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 1].set_xlabel("Epoch")
    canvas.append((fig, axes))

    fig, axes = canvas[0]
    epoch = data["global_step"]
    axes[0, 0].plot(
        epoch,
        data["loss"]["critic"].reshape(-1, data["loss"]["critic"][0].size),
        **kwargs
    )
    axes[0, 1].plot(
        epoch,
        data["loss"]["actor"].reshape(-1, data["loss"]["actor"][0].size),
        **kwargs
    )

    # theta = []
    # for k, v in data["state_dict"].items():
    #     if k.startswith("net_pi") and k.endswith("weight"):
    #         theta.append(v.reshape(-1, v[0].size))
    # theta = np.hstack(theta)
    # theta = np.diff(theta, axis=0)

    # axes[1, 1].plot(epoch[:-1], theta, **kwargs)

    # axes[1, 0].plot(
    #     epoch,
    #     data["v"].reshape(-1, data["v"][0].size),
    #     **kwargs
    # )
    # ln, *_ = axes[1, 1].plot(
    #     epoch,
    #     data["theta"].reshape(
    #         -1, np.multiply(*data["theta"][0].shape)),
    #     **kwargs
    # )
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
