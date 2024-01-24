import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter

from configs import plen

# trans.rew == REW.SUC
# (trans.rew == REW.SUC or trans.rew == REW.HALF_SUC)


def mylog(s, mode='a'):
    with open('log/log.txt', mode) as f:
        f.writelines(s)


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
    # return x


def plot_throughput(ddict, dir, win_s, pkt_len):
    # flag = False
    win_len = int(win_s/9e-6)
    data_len = len(next(iter(ddict['throughput'].values())))
    ones_win = np.ones(win_len)*pkt_len/win_len
    x = np.arange(data_len)*9e-6

    # plt.style.use(['science','ieee'])
    plt.figure()
    for k in ddict['throughput'].keys():
        try:
            y = np.convolve(ddict['throughput'][k], ones_win)[:data_len]
        except:
            print(ddict['throughput'][k])
        plt.plot(x, y, label=k)

    plt.ylim(-0.1, 1.1)
    plt.xlabel('s', loc='right')
    plt.title(f'throughput, win={win_s}s')
    plt.legend(loc='upper left')
    plt.grid()
    plt.savefig(f'{dir}/throughput.png')
    plt.close()

    # return flag


def plot_loss(ddict, dir):
    plt.figure()
    for k in ddict['loss'].keys():
        plt.plot(ddict['loss'][k], label=k)
    plt.title(f'loss')
    plt.legend()
    plt.grid()
    plt.savefig(f'{dir}/loss.png')
    plt.close()

def plot_multi_throughput(npys):
    pass


def plot_p_unk(ddict, dir):
    from scipy.interpolate import interp1d

    plt.figure()
    for k in ddict['p_unk'].keys():
        tmp = np.array(ddict['p_unk'][k])
        # print(tmp.shape)
        x = tmp[:, 0]*9e-6
        y = tmp[:, 1]
        # plt.plot(x, y, label=k)

        xnew = np.linspace(x.min(),x.max(),1000)
        func = interp1d(x,y)
        ynew = func(xnew)
        plt.plot(xnew, ynew, label=k)


        # plt.plot([1, 2, 3], [3,4,5], label="k")

    # plt.plot([[1,2], [3, 3], [5, 3]], label="k")
    # plt.plot([1, 2, 3], [3,4,5], label="k")
    plt.title(f'p_unk')
    plt.legend()
    plt.grid()
    plt.savefig(f'{dir}/p_unk.png')
    plt.close()
