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
@click.argument("expdir", type=click.Path(exists=True))
@click.option("--plot", "-p", is_flag=True)
def run(expdir, plot):
    envparams = dict(dt=10, max_t=100)
    envparams["logging_path"] = os.path.join(expdir, "test.h5")
    env = envs.F16Dof3(**envparams)

    agentparams = dict()
    agent = agents.Agent(
        **agentparams,
        reward_fn=env.reward,
        reward_grad_fn=env.reward_grad
    )

    env.set_inner_ctrl(agent)

    info = dict(envparams=envparams, agentparams=agentparams)
    env.logger.set_info(info)

    print(f"Runnning {info['expname']} ...")

    env.reset()
    while True:
        env.render()
        if env.step()[2]:
            break

    env.close()

    if plot:
        import figures
        import fym.plotting as plotting

        histpath = os.path.join(expdir, "hist.h5")
        testpath = os.path.join(expdir, "test.h5")

        figures.train_plot(histpath)
        data = logging.load(testpath)
        plotter = plotting.Plotter()
        plotter.plot2d(data["time"], data["state"])

        plotter.show()


@main.command()
@click.argument("dirname", type=click.Path(exists=True))
@click.option("--sample", "mode", flag_value="sample")
@click.option("--test", "mode", flag_value="test")
def plot(dirname, mode):
    import figures
    import fym.plotting as plotting

    if mode == "test":
        histpath = os.path.join(dirname, "hist.h5")
        testpath = os.path.join(dirname, "test.h5")

        figures.train_plot(histpath)
        data = logging.load(testpath)
        plotter = plotting.Plotter()
        plotter.plot2d(data["time"], data["state"])

        plotter.show()
    elif mode == "sample":
        samplefiles = common.parse_file(dirname, ext="h5")
        plotter = plotting.Plotter()
        for file in tqdm(samplefiles):
            data = logging.load(file)
            plotter.plot2d(data["time"], data["state"])

        plotter.show()


if __name__ == "__main__":
    main()
