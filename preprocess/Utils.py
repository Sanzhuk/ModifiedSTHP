import math
import torch

def get_count_model_tensor(time_data, event_data, bin_size):
    mx = max(time_data)
    
    L = math.ceil(mx / bin_size)
    K = max(event_data)
    C = torch.zeros(L, K)
    
    for i in range(len(time_data)):
        interval_idx = math.ceil(time_data[i] / bin_size) - 1
        event_idx = event_data[i] - 1
        
        C[interval_idx][event_idx] += 1
        
    return C