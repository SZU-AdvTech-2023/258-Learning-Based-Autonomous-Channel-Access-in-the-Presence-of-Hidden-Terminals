import numpy as np

from configs import topo_dict


class ACT:
    IDLE = 0
    Tx = 1
    RTS = 2


class OBS:
    IDLE = 0
    Tx = 1
    Unk = 2


class REL:
    ME = 0
    OH = 1
    TH = 2


class REW:
    IDLE = 0
    COL = -1
    SUC = 1


class ACK:
    NONE = 0
    ACK = 1
    NAK = -1
    CTS = 2


class MyMultiAgentEnv():
    def __init__(self, args):
        self.topo = topo_dict[args.task][args.topo]
        self.n = len(self.topo)
        self.pkt_len = args.pkt_len
        self.seq_len = args.seq_len
        self.get_throughput = None

        # actor_input_dim, actor_output_dim, critic_input_dim, critic_output_dim
        # self.dim = (3, 2, self.n, self.n)

        self.t = 0
        self.tx_counter_list = np.zeros(self.n, np.uint8)
        self.tx_cache = np.full(self.pkt_len, -1, np.uint8)

    def reset(self):
        self.t = 0
        self.tx_counter_list.fill(0)
        self.tx_cache.fill(-1)
        return np.zeros((self.n, 3), np.float32)

    def step(self, action_list: np.ndarray):
        self.t += 1
        tx_counter = (action_list == ACT.Tx).sum()
        self.tx_cache[:-1] = self.tx_cache[1:]
        if tx_counter == 1:
            self.tx_cache[-1] = action_list.argmax()
        else:
            self.tx_cache[-1] = -1

        for i in range(self.n):
            if action_list[i] == ACT.Tx:
                self.tx_counter_list[i] += 1
            else:
                self.tx_counter_list[i] = 0

        obs = np.zeros((self.n, 3), np.float32)
        rew = np.zeros(self.n, np.float32)
        done = np.zeros(self.n, bool)
        info = {
            'txack': np.zeros(self.n, np.float32),
            'ack': 'non',
            'throughput_list_done': np.ndarray
        }

        obs[:, 0] = action_list

        for i in range(self.n):
            if action_list[i] == ACT.Tx:
                obs[i, 1] = OBS.Unk
            elif ACT.Tx in action_list[self.topo[i] == REL.OH]:
                obs[i, 1] = OBS.Tx
            else:
                obs[i, 1] = OBS.IDLE

        tx_done_index = np.argwhere(self.tx_counter_list == self.pkt_len)
        if tx_done_index.size != 0:
            if (self.tx_cache == tx_done_index[0]).all():
                # assert len(arg) == 1
                info['txack'][tx_done_index[0]] = 1
                obs[:, 2] = ACK.ACK
                info['ack'] = 'ack'
            else:
                obs[:, 2] = ACK.NAK
                info['ack'] = 'nak'
            self.tx_counter_list[tx_done_index] = 0
        else:
            obs[:, 2] = ACK.NONE
            info['ack'] = 'non'

        # # del ACK
        obs[:, 2] = OBS.Unk # turn off if dqn
        # obs[:, 1] = OBS.IDLE

        throughput_list, real_window = self.get_throughput(min(self.seq_len, self.t))
        throughput_list_done = throughput_list.copy()
        if info['ack'] == 'ack':
            throughput_list_done[action_list == ACT.Tx] += 1
        info['throughput_list_done'] = throughput_list_done

        # global Diff1 reward 1117
        if info['ack'] == 'ack':
            if throughput_list_done.max()-throughput_list_done.min() <= 1:
                rew[:] = 1
            elif action_list[throughput_list_done.argmin()] == ACT.Tx:
                rew[:] = 1
            else:
                rew[:] = -1
        elif info['ack'] == 'nak':  # turn off when ai coexist with csma
            rew[:] = -1
        else:
            rew[:] = 0

        rts_counter = (action_list == ACT.RTS).sum()
        if rts_counter == 1:
            info['ack'] = 'cts'
            obs[:, 2] = ACK.CTS
            # print(self.t, "ENV CTS")
        elif rts_counter > 1:
            pass
            # print(self.t, "ENV NO CTS")

        return obs, rew, done, info

    def render(self):
        pass
