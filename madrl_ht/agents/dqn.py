from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import device
from models.bilstm import DQNNet
from multi_agent_env import ACK, ACT, OBS
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from .agent import Actor


class DQNBuffer:
    def __init__(self, obs_dim, act_dim, size, is_morl=False):
        self.obs_buf = np.zeros((size, *obs_dim), np.float32)
        self.act_buf = np.zeros((size, 1), np.int64)
        self.rew_buf = np.zeros((size, 1), np.float32)
        if is_morl:
            self.rew_buf = np.zeros((size, 2), np.float32)
        self.next_obs_buf = np.zeros((size, *obs_dim), np.float32)
        self.ptr, self.max_size = 0, size
        self.is_full = False

    def store(self, obs, act, rew, next_obs):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        # self.next_obs_buf[self.ptr] = next_obs
        self.ptr = (self.ptr + 1) % self.max_size
        if self.ptr == 0:
            self.is_full = True

    def get(self):
        if self.is_full:
            return dict(
                obs=np.delete(self.obs_buf, self.ptr - 1, 0),
                act=np.delete(self.act_buf, self.ptr - 1, 0),
                rew=np.delete(self.rew_buf, self.ptr - 1, 0),
                next_obs=np.roll(np.delete(self.obs_buf, self.ptr, 0), 1),
            )
        elif self.ptr >= 2:
            return dict(
                obs=self.obs_buf[: self.ptr - 1],
                act=self.act_buf[: self.ptr - 1],
                rew=self.rew_buf[: self.ptr - 1],
                next_obs=self.obs_buf[1 : self.ptr],
            )
        else:
            return


class DQNBuffer_deque:
    def __init__(self, size):
        self.deque = deque([], size)

    def store(self, obs, act, rew):
        self.deque.append((obs, act, rew))

    def get(self, n_step=1, gamma=0.9):
        deque_list = list(self.deque)
        if len(deque_list) < 2:
            return
        if n_step == 1:
            return dict(
                obs=np.array([dl[0] for dl in deque_list], np.float32)[:-1],
                act=np.expand_dims(
                    np.array([dl[1] for dl in deque_list], np.int64)[:-1], 1
                ),
                # rew=np.array([dl[2] for dl in deque_list], np.float32)[:-1],  # for morl
                rew=np.array([dl[2] for dl in deque_list], np.float32)[:-1].reshape(
                    -1, 1
                ),
                next_obs=np.array([dl[0] for dl in deque_list], np.float32)[1:],
            )
        elif len(deque_list) >= n_step:
            assert False
            # ones = np.ones(n_step)
            # ones = np.logspace(1, n_step, num=n_step, base=gamma)
            ones = np.logspace(0, n_step - 1, num=n_step, base=gamma)
            rew = np.array([dl[2] for dl in deque_list], np.float32)[:-1]
            # rew1 = np.convolve(rew[:, 0], ones, "vaild")
            # rew2 = np.convolve(rew[:, 1], ones, "vaild")
            # rewx = np.vstack((rew1, rew2)).T
            rewx = np.convolve(rew[:, 0], ones, "vaild").reshape(-1, 1)
            # print(np.array([dl[0] for dl in deque_list], np.float32)[:-n_step].shape, rew.shape, rewx.shape)
            return dict(
                obs=np.array([dl[0] for dl in deque_list], np.float32)[:-n_step],
                act=np.array([dl[1] for dl in deque_list], np.int64)[:-n_step].reshape(
                    -1, 1
                ),
                rew=rewx.astype(np.float32),
                next_obs=np.array([dl[0] for dl in deque_list], np.float32)[
                    1 : len(rewx) + 1
                ],
            )


class RNNSelect(nn.Module):
    def __init__(self, tuple_index, dim, index):
        super().__init__()
        self.tuple_index = tuple_index
        self.dim = dim
        self.index = index

    def forward(self, input):
        return input[self.tuple_index].select(dim=self.dim, index=self.index)


class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        input_dim = int(np.prod(state_shape))
        output_dim = int(np.prod(action_shape))
        hidden_layer_size = 128
        layer_num = 2

        seq_len = 50

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_layer_size),
            nn.GRU(
                input_size=hidden_layer_size,
                hidden_size=hidden_layer_size,
                num_layers=layer_num,
                batch_first=True,
            ),
            RNNSelect(0, 1, -1),
            nn.Linear(hidden_layer_size, hidden_layer_size),
        )
        self.Q = nn.Sequential(
            nn.Linear(hidden_layer_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_dim),
        )
        self.V = nn.Sequential(
            nn.Linear(hidden_layer_size, 512), nn.ReLU(inplace=True), nn.Linear(512, 1)
        )

    def forward(self, obs):
        logits = self.model(obs)
        q = self.Q(logits)
        v = self.V(logits)
        logits = q - q.mean(dim=1, keepdim=True) + v
        # return torch.exp(logits.view(-1, 2, 2))
        return logits


class DQN_Actor(Actor):
    def __init__(self, name, obs_dim, act_dim, args):
        super().__init__(name, obs_dim, act_dim)
        self.name = name
        self.device = args.device
        self.pkt_len = args.pkt_len
        self.is_lbt = args.is_lbt

        self.model = Net(obs_dim, 2).to(device)
        self.model_old = Net(obs_dim, 2).to(device)
        self.model_old.load_state_dict(self.model.state_dict())

        if args.is_morl:
            self.model = Net(obs_dim, 4).to(device)
            self.model_old = Net(obs_dim, 4).to(device)
            self.model_old.load_state_dict(self.model.state_dict())

        self.optim = Adam(self.model.parameters(), lr=args.lr)

        self.buf = DQNBuffer_deque(args.buffer_size)

        self._gamma = args.gamma
        # self.n_step = args.n_step

        self.t = 0
        self.tx_counter = 0

        self.eps = 0.9

        self._freq = args.target_update_freq
        self._iter = 0
        self._is_double = True
        self._clip_loss_grad = False
        self._n_step = args.n_step

        self.is_morl = args.is_morl
        self.loss_lam = 1.0

        self.exp_flag = False
        self.temp_exp = None
        self.ack_flag = False

    def act(self, obs):
        self.t += 1
        self.exp_flag = False
        if self.tx_counter == self.pkt_len:
            self.tx_counter = 0
            if self.is_lbt:
                # if self.is_lbt and self.ack_flag != ACK.ACK:
                # if self.is_lbt and self.ack_flag == ACK.NONE:
                return ACT.IDLE, [-0.123, -0.123]
        if self.tx_counter == 0:
            if self.is_lbt and obs[-1, 1] != OBS.IDLE:
                # if self.is_lbt and (obs[-1, 1] != OBS.IDLE and self.ack_flag != ACK.ACK):
                # if self.is_lbt and (obs[-1, 1] != OBS.IDLE and self.ack_flag == ACK.NONE):
                act, logp_a = ACT.IDLE, [-0.123, -0.123]
            else:
                self.exp_flag = True
                act, logp_a = self.get_act(obs)
        else:
            act, logp_a = ACT.Tx, [0.456, 0.456]  # Must send a complete packet
        if act == ACT.Tx:
            self.tx_counter += 1
        return act, logp_a

    def get_act(self, obs):
        if self.t <= 5000:
            self.eps = max(self.eps * 0.99, 0.05)
        else:
            self.eps = 0.00

        obs = torch.FloatTensor(obs).unsqueeze_(0).to(self.device)
        # if np.random.randn() <= self.eps or self.t > 1000:
        if np.random.uniform() >= self.eps:
            with torch.no_grad():
                act_value = self.model(obs)
            act = act_value.argmax()
            # act_value = act_value.tolist()[0]
            if self.is_morl:
                act = self.get_H_val(act_value).argmax()
                act_value = self.get_H_val(act_value).cpu().numpy()[0]
            else:
                act = act.data
                act_value = act_value.tolist()[0]
        else:
            act = np.random.randint(0, 2)
            act_value = [0.001, 0.001]
        return act, act_value

    def update(self):
        if self._iter % self._freq == 0:
            self.model_old.load_state_dict(self.model.state_dict())

        # data = self.buf.get()
        data = self.buf.get(self._n_step, self._gamma)

        if data is None:
            # print(self.t, self.name, "no exp1")
            return -1

        obs = torch.from_numpy(data["obs"]).to(self.device)
        act = torch.from_numpy(data["act"]).to(self.device)
        rew = torch.from_numpy(data["rew"]).to(self.device)
        next_obs = torch.from_numpy(data["next_obs"]).to(self.device)

        # print(obs.shape, self.eps)
        if obs.shape[0] == 0:
            # print(self.t, self.name, "no expssssssssssssssssssssssssssssss")
            return -1

        if self.is_morl:
            return self.update_morl(obs, act, rew, next_obs)

        q = self.model(obs).gather(1, act)
        q_target = self.model_old(next_obs).detach()

        # self._is_double = False

        if self._is_double:
            # (self._gamma**self._n_step)
            returns = rew + self._gamma * q_target.gather(
                1, self.model(next_obs).argmax(dim=1, keepdim=True)
            )
        else:
            returns = rew + self._gamma * q_target.amax(dim=1, keepdim=True)

        # print(rew.shape, q_target.gather(1, self.model(next_obs).argmax(dim=1, keepdim=True)).shape)
        # print(q.shape, returns.shape)

        if self._clip_loss_grad:
            loss = F.huber_loss(q, returns)
        else:
            loss = F.mse_loss(q, returns)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        # self.scheduler.step()

        self._iter += 1
        return loss.item()

    def update_morl(self, obs, act, rew, next_obs):
        q = self.model(obs).gather(1, act.expand(-1, 2).unsqueeze(1))
        q_target = self.model_old(next_obs).detach()

        # self._is_double = False
        if self._is_double:
            q_target_act = self.get_H_val(self.model(next_obs).detach()).argmax(
                dim=1, keepdim=True
            )
            returns = rew.unsqueeze(1) + self._gamma * q_target.gather(
                1, q_target_act.expand(-1, 2).unsqueeze(1)
            )
        else:
            q_target_act = self.get_H_val(q_target.clone()).argmax(dim=1, keepdim=True)
            returns = rew.unsqueeze(1) + self._gamma * q_target.gather(
                1, q_target_act.expand(-1, 2).unsqueeze(1)
            )

        # loss = F.mse_loss(q, returns)
        self.loss_lam = max(0.01, self.loss_lam * 0.999)
        loss_a = F.mse_loss(q, returns)
        loss_b = F.mse_loss(self.get_H_val(q), self.get_H_val(returns))
        loss = self.loss_lam * loss_a + (1 - self.loss_lam) * loss_b

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        # self.scheduler.step()

        self._iter += 1
        return loss.item()

    def reset(self):
        raise NotImplementedError

    def store(self, obs, act, rew):
        self.buf.store(obs, act, rew)
    def FedAvg(self, model_avg):
        obs = torch.from_numpy(self.buf.get()["obs"]).to(self.device)

        with torch.no_grad():
            # kl = F.kl_div(self.model(obs), self.model(obs), reduction="batchmean")
            # kl = F.kl_div(self.model(obs), model_avg(obs), reduction="batchmean").item()

            kl = F.mse_loss(self.model(obs), model_avg(obs)).item()

        # self.model.load_state_dict(model_avg.state_dict())
        alpha = 1 / (1 + abs(kl))
        # alpha = 1 - 1 / (1 + np.log(1 + abs(kl)))
        print(self.name, "kl", kl, alpha)

        state_dict = self.model.state_dict()
        state_dict_fed = model_avg.state_dict()
        state_dict_sum = deepcopy(state_dict)
        for key in list(self.model.state_dict().keys()):
            state_dict_sum[key] = (1 - alpha) * state_dict[
                key
            ] + alpha * state_dict_fed[key]

        self.model.load_state_dict(state_dict_sum)

    def get_H_val(self, q: torch.Tensor, n=2):
        return (
            q.mul(torch.FloatTensor([1, 1 / n]).to(self.device))
            .log()
            .mul(torch.FloatTensor([1, n]).to(self.device))
            .sum(dim=2)
            .detach()
        )

        return q.log().sum(dim=2).detach()

        # print(q.shape)
        # print(q.mean(dim=2).pow(3).shape)
        # print(q.pow(2).mean(dim=2).shape)
        return q.mean(dim=2).pow(3) / q.pow(2).mean(dim=2)
