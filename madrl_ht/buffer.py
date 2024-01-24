import numpy as np
import torch
from utils import discount_cumsum
from configs import device, history_len


def discount_cumsum2(x, discount, length):
    # gamma=1
    # discount = 2.
    win = np.logspace(0, length-1, length, base=discount)
    return np.convolve(x, win[::-1])[-len(x):]


class PPOBuffer_Actor:
    def __init__(self, obs_dim, act_dim, size, gamma, lam):
        self.obs_buf = np.zeros((size, *obs_dim))
        self.act_buf = np.zeros(size)
        self.adv_buf = np.zeros(size)
        self.rew_buf = np.zeros(size)
        self.ret_buf = np.zeros(size)
        self.val_buf = np.zeros(size)
        self.logp_buf = np.zeros(size)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.max_size = 0, size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1
        # self.ptr = (self.ptr+1) % self.max_size
        # print('self.ptr11111111',self.ptr)

    def finish_path(self):
        # GAE-Lambda
        last_val = self.val_buf[-1]
        rews = np.append(self.rew_buf, last_val)
        vals = np.append(self.val_buf, last_val)
        deltas = rews[:-1]+self.gamma*vals[1:]-vals[:-1]
        self.adv_buf = discount_cumsum(deltas, self.gamma*self.lam).copy()

        self.adv_buf = (self.adv_buf-self.adv_buf.mean())/(self.adv_buf.std()+1e-7)

    def get(self):
        if self.ptr == 0:
            return None
        # assert self.ptr == self.max_size
        # print('self.ptr',self.ptr)
        self.finish_path()
        # data = dict(obs=self.obs_buf, act=self.act_buf, logp=self.logp_buf, adv=self.adv_buf)
        data = dict(obs=self.obs_buf[:self.ptr], act=self.act_buf[:self.ptr], logp=self.logp_buf[:self.ptr], adv=self.adv_buf[:self.ptr])
        # self.ptr = 1
        # data = {'obs': self.obs_buf[:self.ptr], 'act': self.act_buf[:self.ptr], 'logp': self.logp_buf[:self.ptr], 'adv': self.adv_buf[:self.ptr]}
        self.ptr = 0
        return {k: torch.FloatTensor(v).to(device) for k, v in data.items()}


class PPOBuffer_Critic:
    def __init__(self, obs_dim, rew_dim, size, gamma, lam=0.95):
        self.rew_dim = rew_dim
        self.obs_buf = np.zeros((size, *obs_dim))
        self.rew_buf = np.zeros((size, rew_dim))
        self.ret_buf = np.zeros((size, rew_dim))
        self.gamma, self.lam = gamma, lam
        self.ptr, self.max_size = 0, size

    def store(self, obs, rew):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.rew_buf[self.ptr] = rew
        self.ptr += 1
        # self.ptr = (self.ptr+1) % self.max_size

    def finish_path(self, last_val=0):
        # rew_buf = (self.rew_buf[:self.ptr]-self.rew_buf[:self.ptr].mean())/(self.rew_buf[:self.ptr].std()+1e-7)
        # self.ret_buf = discount_cumsum(rew_buf, self.gamma).copy()

        self.ret_buf = discount_cumsum(self.rew_buf[:self.ptr], self.gamma).copy()
        self.ret_buf = (self.ret_buf-self.ret_buf.mean())/(self.ret_buf.std()+1e-7)

    def get(self):
        # assert self.ptr == self.max_size
        self.finish_path()
        # data = dict(obs=self.obs_buf, ret=self.ret_buf)
        # data = dict(obs=self.obs_buf[:self.ptr], ret=self.ret_buf[:self.ptr])
        data = dict(obs=self.obs_buf[:self.ptr], ret=self.ret_buf[:self.ptr], rew=self.rew_buf[:self.ptr])
        self.ptr = 0
        return {k: torch.FloatTensor(v).to(device) for k, v in data.items()}

