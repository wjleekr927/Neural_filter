import numpy as np 

# Exponentially decreasing channel
def channel_gen(total_taps, decay_factor, seed):
    np.random.seed(seed)
    channel_vec = np.zeros((total_taps,1), dtype = 'complex_')
    # Not use exponential here
    # exp_decay = np.round(np.exp(-decay_factor), 4)
    for idx in range(total_taps):
        # Std to variance
        rnd_complex = np.random.randn(scale=1/(2*total_taps)**.5) + np.random.randn(scale=1/(2*total_taps)**.5) * 1j
        channel_vec[idx] = rnd_complex
        # Test
    
    return channel_vec