from copy import deepcopy

import numpy as np

from agents import CHAOS, CSMA_Actor, DQN_Actor, PPO_Actor, TDMA_Actor, CSMA_RCTS_Actor
from configs import topo_dict
from critics import PPO_Critic
from multi_agent_env import ACK, ACT, OBS


class MultiAgentManager():
    def __init__(self, args):
        self.topo = topo_dict[args.task][args.topo]
        self.n = len(self.topo)
        self.pkt_len = args.pkt_len
        self.seq_len = args.seq_len

        actor_obs_dim, actor_act_dim, critic_obs_dim, critic_val_dim = (3, 2, self.n, self.n)
        # actor_obs_dim, actor_act_dim, critic_obs_dim, critic_val_dim = (3, 2, self.n, 1)

        # CTDE
        self.actors = [PPO_Actor(f'PPO_Actor{i}', actor_obs_dim, actor_act_dim, args) for i in range(self.n)]
        #self.actors = [PPO_Actor(f'PPO_Actor0', actor_obs_dim, actor_act_dim, args), CSMA_Actor(f'CSMA_Actor0', actor_obs_dim, actor_act_dim)]
        #self.actors = [CSMA_Actor(f'CSMA_Actor{i}', actor_obs_dim, actor_act_dim) for i in range(self.n)]
        self.critic = PPO_Critic('critic', critic_obs_dim, critic_val_dim, args)
        self.obs_cache = np.zeros((self.seq_len, self.n, actor_obs_dim))
        self.train_info_dict = self.init_train_info_dict()

        self.t = 0

    def act(self, obs_list):
        self.t += 1
        act_list = np.zeros(self.n, dtype=np.float32)
        logp_list = np.zeros(self.n, dtype=np.float32)
        obs_h = np.vstack((self.obs_cache[1:], np.expand_dims(obs_list, 0)))
        for i in range(self.n):
            act_list[i], logp_list[i] = self.actors[i].act(obs_h[:, i, :])
        return act_list, logp_list

    def update(self):
        for i in range(self.n):
            self.train_info_dict['loss'][f'{self.actors[i].name}'].append(self.actors[i].update())
        self.train_info_dict['loss']['critic'].append(self.critic.update())
        # self.train_info_dict['loss']['critic'].append(-1)
        return self.train_info_dict
    
    def update_not(self):
        for i in range(self.n):
            self.train_info_dict['loss'][f'{self.actors[i].name}'].append(-1)
            self.actors[i].buf.get()
        self.train_info_dict['loss']['critic'].append(-1)
        self.critic.buf.get()
        return self.train_info_dict

    def reset(self):
        # self.train_info_dict = self.init_train_info_dict()
        self.t = 0
        self.obs_cache.fill(0)
        for ac in self.actors:
            ac.reset()

    def store(self, obs, act, rew, logp, info):
        # LBT (A1A1A1  A1B1A1B1  ACACAC)
        self.obs_cache = np.vstack((self.obs_cache[1:], obs[np.newaxis, :]))
        critic_obs = self.obs_cache[:, :, 0].reshape((self.seq_len, self.n))  # all agents' actions
        # val = 1
        val = self.critic.get_val(critic_obs)
        self.critic.store(deepcopy(critic_obs), rew)

        store_info = {}

        for i in range(self.n):
            if not isinstance(self.actors[i], PPO_Actor):
                continue
            # PPO vaild exp
            # if act[i] == ACT.IDLE:
            if act[i] == ACT.IDLE and self.obs_cache[-1, i, 1] == OBS.IDLE and self.obs_cache[-1, i, 0] == ACT.IDLE:  # LBT
                self.actors[i].store(deepcopy(self.obs_cache[:, i, :]), act[i], rew[i], val[i], logp[i])
                store_info[i] = 1

            elif act[i] == ACT.Tx:
                if self.actors[i].tx_counter == 1:
                    self.actors[i].temp_exp = {'obs': deepcopy(self.obs_cache[:, i, :]), 'act': act[i], 'rew': 0, 'val': val[i], 'logp': logp[i]}
                    # self.actors[i].temp_exp = {'obs': deepcopy(self.obs_cache[:, i, :]), 'act': act[i], 'rew': 0, 'val': val, 'logp': logp[i]}
                elif self.actors[i].tx_counter == self.pkt_len:
                    exp = self.actors[i].temp_exp
                    exp['rew'] = rew[i]
                    self.actors[i].store(**exp)
                    store_info[i] = 1
                    # store_info[i] = exp

        # record throughput
        self.train_info_dict['throughput']['actors'].append(info['txack'].max())
        for i in range(self.n):
            self.train_info_dict['throughput'][f'{self.actors[i].name}'].append(info['txack'][i])

        return store_info

    def get_throughput(self, window):
        real_window = min(window, len(self.train_info_dict['throughput']['actors']), self.t)
        return np.array([sum(self.train_info_dict['throughput'][f'{self.actors[i].name}'][-real_window:]) for i in range(self.n)], np.float32), real_window

    def init_train_info_dict(self):
        train_info_dict = {'loss': {'critic': []}, 'throughput': {'actors': []}}
        # train_info_dict['value'] = {'af': []}
        for i in range(self.n):
            train_info_dict['loss'][f'{self.actors[i].name}'] = []
            train_info_dict['throughput'][f'{self.actors[i].name}'] = []
        return train_info_dict

    def save_model(self, log_dir):
        self.critic.save_model(log_dir)
        for ac in self.actors:
            ac.save_model(log_dir)

    def load_model(self, log_dir):
        self.critic.load_model(log_dir)
        for ac in self.actors:
            ac.load_model(log_dir)

    def correct_exp_after(self, act, info, next_obs):
        for i in range(self.n):
            if act[i] == ACT.IDLE:
                if next_obs[i, 1] == OBS.IDLE:
                    if info['ack'] == 'non':
                        pass
                    elif info['ack'] == 'ack':
                        if self.pkt_len > 1:
                            self.obs_cache[-(self.pkt_len-1):, i, 2] = OBS.Tx
                        next_obs[i, 2] = OBS.Tx
                    elif info['ack'] == 'nak':
                        if self.pkt_len > 1:
                            self.obs_cache[-(self.pkt_len-1):, i, 2] = OBS.Tx
                        next_obs[i, 2] = OBS.Tx
                elif next_obs[i, 1] == OBS.Tx:
                    if info['ack'] == 'non':
                        pass
                    elif info['ack'] == 'ack':
                        if self.pkt_len > 1:
                            self.obs_cache[-(self.pkt_len-1):, i, 2] = OBS.IDLE
                        next_obs[i, 2] = OBS.IDLE
                    elif info['ack'] == 'nak':
                        pass
            elif act[i] == ACT.Tx:
                if info['ack'] == 'non':
                    pass
                elif info['ack'] == 'ack':
                    if self.pkt_len > 1:
                        self.obs_cache[-(self.pkt_len-1):, i, 1] = OBS.IDLE
                    # obs[i, 1] = OBS.IDLE
                    next_obs[i, 1] = OBS.IDLE
                    if self.pkt_len > 1:
                        self.obs_cache[-(self.pkt_len-1):, i, 2] = OBS.IDLE
                    # obs[i, 2] = OBS.IDLE
                    next_obs[i, 2] = OBS.IDLE
                elif info['ack'] == 'nak':
                    pass
        return next_obs