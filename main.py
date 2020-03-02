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
def sample():
    for i in trange(10):
        _sample(i)


def _sample(i):
    np.random.seed(i)

    envparams = dict(dt=0.01, max_t=10)
    env = envs.F16(
        **envparams,
        logging_path=os.path.join("data", "samples", f"{i:03d}.h5")
    )

    env.reset("random")
    env.logger.set_info(
        env=env.__class__.__name__,
        envparams=envparams,
        K=env.get_parameter(),
        initial_state=env.system.initial_state,
        behavior_param=env.get_parameter()
    )
    while True:
        if env.step()[2]:
            break

    env.close()


@main.command()
@click.argument("sampledir", type=click.Path(exists=True))
@click.option("--max-epoch", "-n", default=100)
@click.option("--batch-size", "-b", default=64)
@click.option("--continue", "-c", is_flag=True)
@click.option("--seed", default=0, type=int)
def train(sampledir, **kwargs):
    np.random.seed(kwargs["seed"])
    torch.manual_seed(kwargs["seed"])

    samplefiles = common.parse_file(sampledir, ext="h5")
    data, info = logging.load(samplefiles[0], with_info=True)

    env = getattr(envs, info["env"])(**info["envparams"])

    agentparams = dict(x_size=7, u_size=2, lr_h=1e-3, lr_g=1e-3, lr_pi=1e-4)
    agent = agents.Agent(
        **agentparams,
        reward_fn=env.reward,
        reward_grad_fn=env.reward_grad
    )

    env.set_inner_ctrl(agent)

    expname = "-".join((env.name, agent.get_name()))
    expdir = os.path.join("data", "train", expname)
    histpath = os.path.join(expdir, "hist.h5")

    if not kwargs["continue"]:
        if os.path.exists(expdir):
            if input(f"Delete \"{expdir}\"? [Y/n]: ") in ["", "Y", "y"]:
                shutil.rmtree(expdir)
        epoch_init, global_step = 0, 0
        mode = "w"
    else:
        weightpath = sorted(glob.glob(os.path.join(expdir, "*.pth")))[-1]
        data = torch.load(weightpath)
        agent.load_weights(data)
        epoch_init = int(data["epoch"]) + 1
        global_step = int(data["global_step"]) + 1
        mode = "r+"

    true_param = env.K
    true_eigvals = env.get_eigvals(true_param)
    info.update(
        agent=agent.__class__.__name__,
        agentparams=agentparams,
        expname=expname,
        click=kwargs,
        true_param=true_param,
        true_eigvals=true_eigvals.real
    )
    logger = logging.Logger(path=histpath, max_len=1, mode=mode)
    logger.set_info(info)

    dataloader = common.get_dataloader(
        samplefiles,
        keys=("state", "action"),
        shuffle=True,
        batch_size=64
    )

    print(f"Training {expname} ...")

    max_global = kwargs["max_epoch"] * len(dataloader) - 1
    logging_interval = int(1e-4 * max_global) or 10
    print_interval = int(1e-2 * max_global) or 10
    save_interval = int(1e-2 * max_global) or 1

    epoch_final = epoch_init + kwargs["max_epoch"]
    t0 = time.time()
    for epoch in range(epoch_init, epoch_final):
        desc = f"Epoch {epoch:2d}/{epoch_final - 1:2d} | Critic"
        rloss = 0
        for n, data in enumerate(tqdm(dataloader, desc=desc, leave=False)):
            agent.set_input(data)
            loss = agent.update_critic()
            rloss = n / (n + 1) * rloss + 1 / (n + 1) * loss
            agent.info["rloss_critic"] = rloss

            if global_step % print_interval == 0:
                msg = "\t".join([
                    f"[step_{global_step:07d}]",
                    agent.get_msg()
                ])
                tqdm.write(msg)

            if global_step % logging_interval == 0 or global_step == max_global:
                logger.record(
                    global_step=global_step,
                    info=agent.info
                )

            global_step += 1

        if rloss < 1000:
            agent.actor_optimizer.zero_grad()
            desc = f"Epoch {epoch:2d}/{epoch_final - 1:2d} | Actor"
            for n, data in enumerate(tqdm(dataloader, desc=desc, leave=False)):
                agent.set_input(data)
                agent.update_actor()
            agent.actor_optimizer.step()

            param = env.get_param()
            eigvals = env.get_eigvals(param)
            logger.record(
                epoch=epoch,
                loss_actor=copy.deepcopy(agent.info["loss_actor"]),
                param=param,
                eigvals=eigvals.real,
            )
            agent.info["max_eigval"] = eigvals.real.max()

        if epoch % save_interval == 0 or epoch == epoch_final - 1:
            savepath = os.path.join(expdir, f"trained-{global_step:07d}.pth")
            agent.save(epoch, global_step, savepath)

    logger.close()

    print(f"elapsed time: {time.time() - t0:5.2f} sec")
    print(f"exp. saved in \"{expdir}\"")


@main.command()
@click.argument("expdir", type=click.Path(exists=True))
@click.option("--plot", "-p", is_flag=True)
def test(expdir, plot):
    histpath = os.path.join(expdir, "hist.h5")
    data, info = logging.load(histpath, with_info=True)
    weightpath = sorted(glob.glob(os.path.join(expdir, "*.pth")))[-1]
    weight = torch.load(weightpath)

    envparams = info["envparams"]
    envparams["logging_path"] = os.path.join(expdir, "test.h5")
    env = getattr(envs, info["env"])(**envparams)

    agentparams = info["agentparams"]
    agent = getattr(agents, info["agent"])(
        **agentparams,
        reward_fn=env.reward,
        reward_grad_fn=env.reward_grad
    )
    agent.load_weights(weight)
    agent.eval()

    env.set_inner_ctrl(agent)

    Ac = env.system.A + env.system.B.dot(env.get_param())
    eigvals = np.linalg.eigvals(Ac)
    # print(f"Weight: {agent.net_pi.state_dict()['model.0.weight'].numpy()}")
    # print(f"Weight: {env.K}")
    print(f"Eigenvalues: {eigvals}")
    info.update(
        envparams=envparams,
        agentparams=agentparams,
        weightpath=weightpath,
    )
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
