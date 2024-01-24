import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from buffer import PPOBuffer_Actor, PPOBuffer_Critic
from models.bilstm import ActorNet, CriticNet
from multi_agent_env import ACT, OBS
from torch.distributions.categorical import Categorical
from torch.optim import Adam, RAdam
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from utils import alpha_func, discount_cumsum
from torch.optim.lr_scheduler import CosineAnnealingLR

from .agent import Actor


class PPO_Actor_Buffer_new:
    def __init__(self, obs_dim, act_dim, size, history_len, gamma=0.9, lam=0.95):
        self.obs_buf = np.zeros((size+history_len-1, obs_dim))
        self.act_buf = np.zeros(size+history_len)
        self.adv_buf = np.zeros(size)
        self.rew_buf = np.zeros(size+history_len)
        self.ret_buf = np.zeros(size)
        self.val_buf = np.zeros(size+history_len)
        self.logp_buf = np.zeros(size+history_len)
        self.txcounter_buf = np.zeros(size+history_len)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.max_size = history_len-1, size+history_len-1

    def store(self, obs, act, rew, val, logp, txcounter):
        # def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.txcounter_buf[self.ptr] = txcounter
        self.ptr += 1

    def finish_path(self):
        # rew_buf = (self.rew_buf[:self.ptr]-self.rew_buf[:self.ptr].mean())/(self.rew_buf[:self.ptr].std()+1e-7)
        # self.ret_buf = discount_cumsum(rew_buf, self.gamma).copy()

        self.ret_buf = discount_cumsum(self.rew_buf[history_len-1:self.ptr], self.gamma).copy()
        self.ret_buf = (self.ret_buf-self.ret_buf.mean())/(self.ret_buf.std()+1e-7)
        self.adv_buf = self.ret_buf-self.val_buf[history_len-1:self.ptr]
    def get(self):
        # assert self.ptr == self.max_size
        # print('self.ptr',self.ptr)
        self.finish_path()
        # data = dict(obs=self.obs_buf, act=self.act_buf, logp=self.logp_buf, adv=self.adv_buf)
        # data = dict(obs=self.obs_buf[:self.ptr], act=self.act_buf[:self.ptr], logp=self.logp_buf[:self.ptr], adv=self.adv_buf[:self.ptr])
        # data = dict(obs=np.array([self.obs_buf[i:i+history_len] for i in range(self.ptr-history_len+1)]), act=self.act_buf[history_len-1:self.ptr].copy(), logp=self.logp_buf[history_len-1:self.ptr].copy(), adv=self.adv_buf)

        # print((self.txcounter_buf[history_len-1:self.ptr] == 0).sum()+(self.txcounter_buf[history_len-1:self.ptr] == plen).sum())
        vaild_pos = (self.txcounter_buf[history_len-1:self.ptr] <= 1)
        vaild_sum = (self.txcounter_buf[history_len-1:self.ptr] <= 1).sum()
        if self.txcounter_buf[-1] not in [0, plen]:
            vaild_sum -= 1
        # 保证要成对出现
        # print(vaild_pos)
        data = dict(obs=self.obs_buf[np.array([np.arange(i, i+history_len) for i in range(self.ptr-history_len+1)])][vaild_pos][:vaild_sum],
                    act=self.act_buf[history_len-1:self.ptr].copy()[vaild_pos],
                    logp=self.logp_buf[history_len-1:self.ptr].copy()[vaild_pos],
                    adv=self.adv_buf[vaild_pos])
        # a[np.array([np.arange(i,i+2) for i in range(4)])]

        # data = {'obs': self.obs_buf[:self.ptr], 'act': self.act_buf[:self.ptr], 'logp': self.logp_buf[:self.ptr], 'adv': self.adv_buf[:self.ptr]}
        self.obs_buf[:history_len] = self.obs_buf[self.ptr-history_len+1:self.ptr+1]
        # self.obs_buf[:history_len] = self.obs_buf[self.ptr-history_len:self.ptr]
        self.act_buf[:history_len] = self.act_buf[self.ptr-history_len+1:self.ptr+1]
        self.rew_buf[:history_len] = self.rew_buf[self.ptr-history_len+1:self.ptr+1]
        self.val_buf[:history_len] = self.val_buf[self.ptr-history_len+1:self.ptr+1]
        self.logp_buf[:history_len] = self.logp_buf[self.ptr-history_len+1:self.ptr+1]
        self.ptr = history_len-1

        return {k: torch.FloatTensor(v).to(device) for k, v in data.items()}


class PPO_Actor(Actor):
    def __init__(self, name, obs_dim, act_dim, args):
        super().__init__(name, obs_dim, act_dim)
        self.name = name
        self.device = args.device
        self.pkt_len = args.pkt_len

        self.pi = ActorNet(obs_dim, 64, 1, act_dim).to(self.device)
        self.pi_optimizer = Adam(self.pi.parameters(), lr=args.pi_lr)
        self.train_pi_iters = args.train_pi_iters
        self.clip_ratio = args.clip_ratio
        self.buf = PPOBuffer_Actor((args.seq_len, obs_dim), act_dim, args.buffer_size, args.gamma, args.gae_lambda)
        self.ent_coef = args.ent_coef

        self.t = 0
        self.tx_counter = 0
        self.temp_exp = {}

        self.p_unk = []

        # CosineAnnealingLR
        # self.pi_scheduler = CosineAnnealingLR(self.pi_optimizer, 50)

    def act(self, obs):
        self.t += 1
        if self.tx_counter == self.pkt_len:
            self.tx_counter = 0
            return ACT.IDLE, np.log(1.0)  # LBT self must gap 1
        if self.tx_counter == 0:
            # act, logp_a = self.get_act(obs)
            if obs[-1, 1] == OBS.IDLE:
                act, logp_a = self.get_act(obs)
            else:
                act, logp_a = ACT.IDLE, np.log(1.0)  # LBT
        else:
            act, logp_a = ACT.Tx, np.log(1.0)  # Must send a complete packet
        if act == ACT.Tx:
            self.tx_counter += 1
        return act, logp_a

    def get_act(self, obs):
        obs = torch.FloatTensor(obs).unsqueeze_(0).to(self.device)
        with torch.no_grad():
            action_probs = self.pi(obs)

        dist = Categorical(action_probs)
        act = dist.sample()
        logp_a = dist.log_prob(act)
        return act.item(), logp_a.item()

    def update(self):
        data = self.buf.get()
        if data is None:
            return -1
        obs, act, logp_old, adv = data['obs'], data['act'], data['logp'], data['adv']

        # # # batch size = allllllllll
        loss_list = []
        for _ in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            action_probs = self.pi(obs)
            dist = Categorical(action_probs)
            logp = dist.log_prob(act)
            ratio = torch.exp(logp-logp_old)
            clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio)*adv
            # loss_pi = -(torch.min(ratio*adv, clip_adv)).mean()
            loss_pi = -(torch.min(ratio*adv, clip_adv)).mean()-self.ent_coef*dist.entropy().mean()
            loss_pi.backward()
            self.pi_optimizer.step()
            loss_list.append(loss_pi.item())


        return np.mean(loss_list)
        # return loss_pi.item()

    def reset(self):
        self.t = 0
        self.tx_counter = 0

    def store(self, obs, act, rew, val, logp):
        # print(self.t, self.name, (obs[:, 1:] == OBS.Unk).sum()/200)
        self.p_unk.append([self.t, (obs[:, 1:] == OBS.Unk).sum()/(100*2)])

        self.buf.store(obs, act, rew, val, logp)

    def save_model(self, log_dir):
        torch.save(self.pi, f'{log_dir}/{self.name}.pt')

    def load_model(self, log_dir, name=None):
        if name:
            self.pi = torch.load(f'{log_dir}/{name}.pt', map_location=self.device)
        else:
            self.pi = torch.load(f'{log_dir}/{self.name}.pt', map_location=self.device)
