import numpy as np 

# Exponentially decreasing channel
def channel_gen(total_taps, decay_factor, seed):
    np.random.seed(seed)
    channel_vec = np.zeros((total_taps,1), dtype = 'complex_')
    exp_decay = np.round(np.exp(-decay_factor), 4)

    for idx in range(total_taps):
        rnd_complex = np.random.randn() + np.random.randn() * 1j
        channel_vec[idx] = exp_decay * rnd_complex

    return channel_vec