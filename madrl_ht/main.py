import argparse
import logging
import os
import shutil
import time

import numpy as np
import torch
from tqdm import tqdm

from multi_agent_env import MyMultiAgentEnv
from multi_agent_manager import MultiAgentManager
from utils import plot_loss, plot_p_unk, plot_throughput


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--task", type=str, default="3agent")
    parser.add_argument("--topo", type=str, default="agent3_topo_3")
    parser.add_argument("--index", type=int, default=99)
    parser.add_argument("--pkt-len", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--is-lbt", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    parser.add_argument("--epoch", type=int, default=3000)
    parser.add_argument("--step-per-epoch", type=int, default=100)
    parser.add_argument("--buffer-size", type=int, default=100)

    # ppo special
    parser.add_argument("--pi-lr", type=float, default=1e-3)
    parser.add_argument("--vf-lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--train-pi-iters", type=int, default=2)
    parser.add_argument("--train-vf-iters", type=int, default=2)
    parser.add_argument("--ent-coef", type=float, default=0.01)

    # plot
    parser.add_argument("--plot-smooth-window", type=float, default=0.01)

    args = parser.parse_known_args()[0]
    return args


def main(args=get_args()):
    # earlystop = False

    log_dir = os.path.join(
        args.logdir,
        args.task,
        args.topo + f"_plen{args.pkt_len}",
        str(args.index),
    )
    shutil.rmtree(log_dir, True)
    os.makedirs(log_dir)

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # init
    logging.basicConfig(
        filename=f"{log_dir}/log.txt",
        filemode="w",
        format="%(message)s",
        level=logging.INFO,
        force=True,
    )
    logger = logging.getLogger()
    logger.info(time.asctime())
    logger.info(args)

    env = MyMultiAgentEnv(args)
    agents = MultiAgentManager(args)
    env.get_throughput = agents.get_throughput
    obs = env.reset()

    for epoch in tqdm(range(1, 1 + args.epoch)):

        for t in range(1, 1 + args.step_per_epoch):
            act, logp = agents.act(obs)
            next_obs, rew, done, info = env.step(act)
            # obs, next_obs = agents.correct_exp(obs, act, info, next_obs)
            kk = agents.store(obs, act, rew, logp, info)
            logger.info(
                ", ".join(
                    [
                        f"{agents.t:<5}",
                        f"obs:{obs.astype(int).tolist()}",
                        f"act:{act.astype(int).tolist()}",

                        f'prob:{np.array2string(np.exp(logp),formatter={"all":lambda x:f"{x:<5}"[:5]})}',
                        f'info_throughput:{info["throughput_list_done"].astype(int).tolist()}',
                        f"rew:{rew.tolist()}",
                        f'{info["ack"]}',
                        f"{list(kk.keys())}",
                    ]
                )
            )
            next_obs = agents.correct_exp_after(act, info, next_obs)
            obs = next_obs

            # # CSMA #
            # mm = {"ack": 1, "non": 0, "nak": 0, "cts": 2}
            # obs[:, 2] = mm[info["ack"]]

        # exit()
        agents.update()
        if epoch % 100 == 0:
            plot_throughput(
                agents.train_info_dict, log_dir, args.plot_smooth_window, args.pkt_len
            )
            plot_loss(agents.train_info_dict, log_dir)

        if epoch % 2000 == 0:
            np.save(f"{log_dir}/train_info_dict.npy", agents.train_info_dict)
            np.save(f"{log_dir}/p_unk.npy", {ac.name: ac.p_unk for ac in agents.actors})

    plot_throughput(
        agents.train_info_dict, log_dir, args.plot_smooth_window, args.pkt_len
    )
    plot_loss(agents.train_info_dict, log_dir)
    np.save(f"{log_dir}/train_info_dict.npy", agents.train_info_dict)

if __name__ == "__main__":
    args = get_args()

    for i in range(1, 2):
        args.index = i
        args.seed = i
        print(args)
        main(args)
