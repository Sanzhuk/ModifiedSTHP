import numpy as np
import torch
import torch.utils.data

from transformer import Constants
import preprocess.Utils as Utils

class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self, data):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """
        self.time = [[elem['time_since_start'] for elem in inst] for inst in data] # 1 x n_samples
        self.time_gap = [[elem['time_since_last_event'] for elem in inst] for inst in data] # 1 x n_samples
        # plus 1 since there could be event type 0, but we use 0 as padding
        self.event_type = [[elem['type_event'] + 1 for elem in inst] for inst in data] # 1 x n_samples
        
        self.length = len(data)
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        return self.time[idx], self.time_gap[idx], self.event_type[idx]

class CountModelData(torch.utils.data.Dataset):
    """ Count Model Data """
    def __init__(self, data, bin_size):
        self.time = [[elem['time_since_start'] for elem in inst] for inst in data] # 1 x n_samples
        self.time_gap = [[elem['time_since_last_event'] for elem in inst] for inst in data] # 1 x n_samples
        # plus 1 since there could be event type 0, but we use 0 as padding
        self.event_type = [[elem['type_event'] + 1 for elem in inst] for inst in data] # 1 x n_samples
        
        self.C = Utils.get_count_model_tensor(time_data=self.time[0], event_data=self.event_type[0], bin_size=bin_size)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return self.C[index, :]
    
    
def pad_time(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])
    
    return torch.tensor(batch_seq, dtype=torch.float32)


def pad_type(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.long)


def collate_fn(insts):
    """ Collate function, as required by PyTorch. """

    time, time_gap, event_type = list(zip(*insts))
    time = pad_time(time)
    time_gap = pad_time(time_gap)
    event_type = pad_type(event_type)
    return time, time_gap, event_type


def get_dataloader(data, batch_size, shuffle=True):
    """ Prepare dataloader. """

    ds = EventData(data)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
    )
    return dl

def get_dataloader_count_model(data, batch_size, bin_size=5, shuffle=False):
    ds = CountModelData(data, bin_size=bin_size)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
    )
    return dl
