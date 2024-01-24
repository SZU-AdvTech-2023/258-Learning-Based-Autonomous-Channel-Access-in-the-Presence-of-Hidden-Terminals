import random

import numpy as np
from configs import plen
from multi_agent_env import ACK, ACT, OBS

from .agent import Actor

# CW0 = 32
# CWmax = 1024
# DIFS_length = 4
# plen = 120

# Q
# CW0 = 2
# CWmax = 128
# DIFS_length = 1
# plen = 5

# W
CW0 = 16
CWmax = 512
DIFS_length = 1
plen = 5


class CsmaStatus:
    idle = 0
    difs = 1
    contention = 2
    hold = 3
    transmission = 4


class CSMA_Actor(Actor):
    def __init__(self, name, obs_dim, act_dim):
        super().__init__(name, obs_dim, act_dim)
        self.name = name
        self.t = 0
        # self.tx_counter = 0

        self.last_action = CsmaStatus.idle
        self.action = CsmaStatus.idle

        self.CW = CW0
        self.DIFS_timer = DIFS_length
        self.backoff_timer = random.randint(1, self.CW)
        self.transmission_timer = plen
        self.is_txend = False

    @property
    def tx_counter(self):
        return plen-self.transmission_timer
        raise NotImplemented

    @property
    def action_info(self):
        if self.action == CsmaStatus.idle:
            return 'idle'
        elif self.action == CsmaStatus.difs:
            return f'difs {self.DIFS_timer}'
        elif self.action == CsmaStatus.contention:
            return f'contention {self.backoff_timer} <{self.CW}>'
        elif self.action == CsmaStatus.hold:
            return 'hold'
        elif self.action == CsmaStatus.transmission:
            return f'transmission {self.tx_counter}'

    def act(self, obs):
        self.t += 1
        # oh_obs, ack, _ = obs[-1]
        _, oh_obs, ack = obs[-1]
        # isbusy = oh_obs == OBS.Tx  # or ack == ACK.ACK

        # if self.last_action == CsmaStatus.transmission and self.transmission_timer == 0:
        if self.is_txend:
            self.is_txend = False
            # self.transmission_timer = plen
            # print(self.name, self.t)
            if ack == ACK.ACK:
                # print(self.name, 'ack')
                self.CW = CW0
                isbusy = False
            else:
                # print(self.name, 'nak')
                if self.CW < CWmax:
                    self.CW = 2*self.CW
                isbusy = oh_obs == OBS.Tx
            self.backoff_timer = random.randint(1, self.CW)
            # print(self.name, 'backoff_timer', self.backoff_timer)
        else:
            isbusy = oh_obs == OBS.Tx

        if self.last_action == CsmaStatus.idle:
            if True:  # if has packet
                self.action = CsmaStatus.difs
            else:
                self.action = CsmaStatus.idle
        elif self.last_action == CsmaStatus.difs:
            if not isbusy:
                if self.DIFS_timer > 0:
                    self.action = CsmaStatus.difs
                else:
                    self.DIFS_timer = DIFS_length
                    if self.backoff_timer == 0:
                        self.action = CsmaStatus.transmission
                    else:
                        self.action = CsmaStatus.contention
            else:
                self.DIFS_timer = DIFS_length
                self.action = CsmaStatus.difs
        elif self.last_action == CsmaStatus.contention:
            if not isbusy:
                if self.backoff_timer > 0:
                    self.action = CsmaStatus.contention
                else:
                    self.action = CsmaStatus.transmission
            else:
                self.action = CsmaStatus.hold
        elif self.last_action == CsmaStatus.hold:
            if not isbusy:
                self.action = CsmaStatus.difs
            else:
                self.action = CsmaStatus.hold
        elif self.last_action == CsmaStatus.transmission:
            if self.transmission_timer > 0:
                self.action = CsmaStatus.transmission
            else:
                # set in ack
                self.transmission_timer = plen
                if True:  # if has packet
                    self.action = CsmaStatus.difs
                else:
                    self.action = CsmaStatus.idle

        if self.action == CsmaStatus.idle:
            pass
        elif self.action == CsmaStatus.difs:
            self.DIFS_timer -= 1
        elif self.action == CsmaStatus.contention:
            self.backoff_timer -= 1
        elif self.action == CsmaStatus.hold:
            pass
        elif self.action == CsmaStatus.transmission:
            self.transmission_timer -= 1
            if self.transmission_timer == 0:
                # return 'TxEnd'
                # self.myack(slot)
                self.is_txend = True

        self.last_action = self.action

        if self.action == CsmaStatus.transmission:
            # print(self.name, 'ACT.Tx')
            return ACT.Tx, np.log(1)
        else:
            return ACT.IDLE, np.log(1)

    def get_act(self, obs):
        pass

    def update(self):
        return -1

    def reset(self):
        # pass
        self.t = 0

    def store(self, obs, act, rew, val, logp):
        pass
