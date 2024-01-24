import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.io import savemat

# plen = 5
# agents_n = 2

def plot_throughput(ddict, win_s, pkt_len):
    win_len = int(win_s/9e-6)
    data_len = len(next(iter(ddict['throughput'].values())))
    ones_win = np.ones(win_len)*pkt_len/win_len
    x = (np.arange(data_len)+1)*9e-6

    sum_plot = ...
    for k in ddict['throughput'].keys():
        print(k)
        if k == 'actors':
            k_label = 'SUM'
            # plt.plot(x, np.convolve(ddict['throughput'][k], ones_win)[:data_len], label=k_label, color='#228833')
            #plt.plot(x[:int(4/9e-6)], np.convolve(ddict['throughput'][k], ones_win)[:int(4/9e-6)], label=k_label, color='#228833')
            plt.plot(x, np.convolve(ddict['throughput'][k], ones_win)[:data_len], label=k_label, color='#228833')
        else: 
            # k_label = '$TEAM\\_'+k[-1]+'$'
            # k_label = f'TEAM$^{k[-1]}$'
            # k_label = f'CSMA$^{k[-1]}$'
            if k[0] == 'P':
                k_label = f'TEAM'
            else:
                k_label = f'CSMA'
            # plt.plot(x, np.convolve(ddict['throughput'][k], ones_win)[:data_len], label=k_label)
            #plt.plot(x[:int(4/9e-6)], np.convolve(ddict['throughput'][k], ones_win)[:int(4/9e-6)], label=k_label)
            plt.plot(x, np.convolve(ddict['throughput'][k], ones_win)[:data_len], label=k_label)

    plt.xlim(left=0)
    plt.ylim(-0.05, 1.05)
    # plt.xlabel('s', loc='right')
    # plt.title(f'Throughput')
    # plt.legend(loc='upper left')
    # plt.legend(ncol=3)
    plt.legend()
    plt.grid()
    #plt.savefig(f'{dir}/throughput_sci.png')
    plt.savefig('throughput_sci2.png')
    plt.close()
    # return throughput_mat

test_name = 'log/2agent/agent2_topo_2_plen5/1'
#throughput_mat = plot_throughput(np.load(f'{test_name}/train_info_dict.npy', allow_pickle=True).item(), test_name, 0.01, 5)
throughput_mat = plot_throughput(np.load(f'{test_name}/train_info_dict.npy', allow_pickle=True).item(), 0.01, 5)
