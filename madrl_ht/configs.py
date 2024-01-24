import numpy as np
import torch

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# simulation time
# epochs = 8000
epochs = ...
local_steps_per_epoch = ...
# local_steps_per_epoch = 200

# packet length
plen = ...
# plen = 120

# historical experience length used to update the network
# look back window
history_len = ...
# history_len = 80

# maximum size of experience pool
# max_buf_size = local_steps_per_epoch-history_len+1
max_buf_size = ...
# max_buf_size = 50

# topo must be array
agent1_topo = np.array([
    [0]
])

agent2_topo_1 = np.array([
    [0, 1],
    [1, 0]
])

agent2_topo_2 = np.array([
    [0, 2],
    [2, 0]
])

# {ABC}
agent3_topo_1 = np.array([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])

# {AB}, {BC}
agent3_topo_2 = np.array([
    [0, 1, 2],
    [1, 0, 1],
    [2, 1, 0]
])

# {AB}, {C}
agent3_topo_3 = np.array([
    [0, 1, 2],
    [1, 0, 2],
    [2, 2, 0]
])

# {A}, {B}, {C}
agent3_topo_4 = np.array([
    [0, 2, 2],
    [2, 0, 2],
    [2, 2, 0]
])

# {ABCD}
agent4_topo_1 = np.array([
    [0, 1, 1, 1],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [1, 1, 1, 0]
])

# {ABC}, {BCD}
agent4_topo_2 = np.array([
    [0, 1, 1, 2],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [2, 1, 1, 0]
])

# {ABC}, {CD}
agent4_topo_3 = np.array([
    [0, 1, 1, 2],
    [1, 0, 1, 2],
    [1, 1, 0, 1],
    [2, 2, 1, 0]
])

# {ABC}, {D}
agent4_topo_4 = np.array([
    [0, 1, 1, 2],
    [1, 0, 1, 2],
    [1, 1, 0, 2],
    [2, 2, 2, 0]
])

# {AB}, {BC}, {CD}, {DA}
agent4_topo_5 = np.array([
    [0, 1, 2, 1],
    [1, 0, 1, 2],
    [2, 1, 0, 1],
    [1, 2, 1, 0]
])

# {AB}, {BC}, {CD}
agent4_topo_6 = np.array([
    [0, 1, 2, 2],
    [1, 0, 1, 2],
    [2, 1, 0, 1],
    [2, 2, 1, 0]
])

# {AB}, {BC}, {D}
agent4_topo_7 = np.array([
    [0, 1, 2, 2],
    [1, 0, 1, 2],
    [2, 1, 0, 1],
    [2, 2, 1, 0]
])

# {AB}, {CD}
agent4_topo_8 = np.array([
    [0, 1, 2, 2],
    [1, 0, 2, 2],
    [2, 2, 0, 1],
    [2, 2, 1, 0]
])

# {AB}, {C}, {D}
agent4_topo_9 = np.array([
    [0, 1, 2, 2],
    [1, 0, 2, 2],
    [2, 2, 0, 2],
    [2, 2, 2, 0]
])

# {A}, {B}, {C}, {D}
agent4_topo_10 = np.array([
    [0, 1, 2, 2],
    [1, 0, 2, 2],
    [2, 2, 0, 2],
    [2, 2, 2, 0]
])

# {ABCDE}
agent5_topo_1 = np.array([
    [0, 1, 1, 1, 1],
    [1, 0, 1, 1, 1],
    [1, 1, 0, 1, 1],
    [1, 1, 1, 0, 1],
    [1, 1, 1, 1, 0]
])

# {ABC}, {DE}
agent5_topo_q = np.array([
    [0, 1, 1, 2, 2],
    [1, 0, 1, 2, 2],
    [1, 1, 0, 2, 2],
    [2, 2, 2, 0, 1],
    [2, 2, 2, 1, 0]
])

# {ABCD}, {E}
agent5_topo_w = np.array([
    [0, 1, 1, 1, 2],
    [1, 0, 1, 1, 2],
    [1, 1, 0, 1, 2],
    [1, 1, 1, 0, 2],
    [2, 2, 2, 2, 0]
])

# {AB}, {CD}, {E}
agent5_topo_e = np.array([
    [0, 1, 2, 2, 2],
    [1, 0, 2, 2, 2],
    [2, 2, 0, 1, 2],
    [2, 2, 1, 0, 2],
    [2, 2, 2, 2, 0]
])

agent5_topo_TH = np.array([
    [0, 2, 2, 2, 2],
    [2, 0, 1, 2, 2],
    [2, 1, 0, 2, 2],
    [2, 2, 2, 0, 1],
    [2, 2, 2, 1, 0]
])

agent6_topo_1 = np.array([
    [0, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 1, 1],
    [1, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 0]
])

agent7_topo = np.array([
    [0, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 1, 1, 1],
    [1, 1, 0, 1, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 0]
])

# topo = agent1_topo
# topo = agent2_topo_2
topo = agent3_topo_3
# topo = agent4_topo_8
# topo = agent5_topo_1
# topo = agent7_topo

topo_dict = {
    '2agent': {
        'agent2_topo_1': agent2_topo_1,
        'agent2_topo_2': agent2_topo_2,
    },
    '3agent': {
        'agent3_topo_1': agent3_topo_1,
        'agent3_topo_2': agent3_topo_2,
        'agent3_topo_3': agent3_topo_3
    },
    '4agent': {
        'agent4_topo_1': agent4_topo_1,
        'agent4_topo_2': agent4_topo_2,
        'agent4_topo_3': agent4_topo_3,
        'agent4_topo_4': agent4_topo_4,
        'agent4_topo_5': agent4_topo_5,
        'agent4_topo_6': agent4_topo_6,
        'agent4_topo_7': agent4_topo_7,
        'agent4_topo_8': agent4_topo_8,
        'agent4_topo_9': agent4_topo_9,
        'agent4_topo_10': agent4_topo_10
    },
    '5agent': {
        'agent5_topo_1': agent5_topo_1,
        'agent5_topo_q': agent5_topo_q,
        'agent5_topo_w': agent5_topo_w
    },
    '6agent': {
        'agent6_topo_1': agent6_topo_1
    }
}
