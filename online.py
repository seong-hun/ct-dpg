import shutil
import os
import glob
import time
import copy
import numpy as np
import click
from tqdm import tqdm, trange

import torch

import fym.logging as logging

import envs
import agents
import common


@click.group()
def main():
    pass


@main.command()
@click.option("--plot", "-p", is_flag=True)
@click.option("--seed", default=1, type=int)
@click.option("--turnon", default=200)
@click.option("--maxt", default=250)
def run(plot, seed, **kwargs):
    np.random.seed(seed)

    expdir = os.path.join("data", "F16Dof3")
    envparams = dict(turnon=kwargs["turnon"],
                     dt=30, max_t=kwargs["maxt"],
                     solver="odeint", ode_step_len=3000)
    envparams["logging_path"] = os.path.join(expdir, "run.h5")
    env = envs.F16Dof3(**envparams)

    info = dict(
        envparams=envparams,
        true_w1=env.true_w1,
        true_w2=env.true_w2,
    )
    env.logger.set_info(info)

    print(f"Runnning {expdir} ...")

    env.reset(mode="random")
    while True:
        env.render()
        if env.step()[2]:
            break

    env.close()

    if plot:
        import figures
        import fym.plotting as plotting

        runpath = os.path.join(expdir, "run.h5")
        figures.plot_online(runpath)
        figures.show()


@main.command()
@click.argument("dirname", type=click.Path(exists=True))
def plot(dirname):
    import figures
    import fym.plotting as plotting

    runpath = os.path.join(dirname, "run.h5")
    figures.plot_online(runpath)
    figures.show()


if __name__ == "__main__":
    main()
