from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import scipy


pkt_len = 5

test_name = 'log/2agent/agent2_topo_2_plen5/1'
ddict = np.load(f'{test_name}/train_info_dict.npy', allow_pickle=True).item()

ind = np.arange(8)

labels = ['0-0.5s', '0.5-1s', '1-1.5s', '1.5-2s', '2-2.5s', '2.5-3s', '3-3.5s', '3.5-4s']

fig, ax = plt.subplots()
ax.tick_params(axis='x', which='minor', bottom=False, top=False)

win_s = 0.01
win_len = int(win_s/9e-6)
data_len = len(next(iter(ddict['throughput'].values())))
ones_win = np.ones(win_len)*pkt_len/win_len
bottom = np.array([0]*8, float)
for k in ddict['throughput'].keys():
    # for k in ['PPO_Actor1', 'PPO_Actor0']:
    k_label = ...
    if k == 'actors':
        continue
    else:
        if k[0] == 'C':
            k_label = f'CSMA'
        else:
            k_label = f'TEAM'

    a = np.convolve(ddict['throughput'][k], ones_win)[:data_len][:int(4/9e-6)]
    b = np.array([np.mean(qqq) for qqq in np.array_split(a, 8)])
    print(k_label, b)

    p = ax.bar(ind, b, bottom=bottom, label=k_label)
    ax.bar_label(p, fmt='%.2f', label_type='center')
    bottom += b

ax.set_xticks(ind, labels)
ax.set_ylim(0, 1)
ax.legend(loc='upper left')
ax.grid(axis='y')

plt.savefig(f'{test_name}/throughputbar.png')
# plt.show()
