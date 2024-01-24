import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from buffer import PPOBuffer_Actor, PPOBuffer_Critic
from configs import device, history_len, max_buf_size
from models.bilstm import ActorNet, CriticNet
from torch.optim import Adam
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from utils import alpha_func, discount_cumsum

from .critic import Critic


class PPO_Critic(Critic):
    # def __init__(self, name, obs_dim, val_dim, gamma=0.99, vf_lr=1e-3, train_v_iters=1):
    def __init__(self, name, obs_dim, val_dim, args):
        self.name = name
        self.device = args.device

        self.vf = CriticNet(obs_dim, 64, 1, val_dim).to(self.device)
        self.vf_optimizer = Adam(self.vf.parameters(), lr=args.vf_lr)
        self.train_vf_iters = args.train_vf_iters

        self.buf = PPOBuffer_Critic((args.seq_len, obs_dim), val_dim, args.buffer_size, args.gamma)

    def update(self):
        data = self.buf.get()
        obs, ret = data['obs'], data['ret']
        if len(obs) == 0:
            print('len(obs) == 0')
            return -1

        loss_list = []
        for _ in range(self.train_vf_iters):
            self.vf_optimizer.zero_grad()
            val = self.vf(obs)
            loss_v = F.mse_loss(val, ret).mean()
            loss_v.backward()
            self.vf_optimizer.step()
            loss_list.append(loss_v.item())

        return np.mean(loss_list)

    def get_val(self, obs):
        with torch.no_grad():
            return self.vf(torch.FloatTensor(obs).unsqueeze_(0).to(device)).squeeze_(0).cpu().numpy()

    def store(self, obs, rew):
        self.buf.store(obs, rew)

    def save_model(self, log_dir):
        torch.save(self.vf, f'{log_dir}/{self.name}.pt')

    def load_model(self, log_dir):
        self.pi = torch.load(f'{log_dir}/{self.name}.pt', map_location=device)
