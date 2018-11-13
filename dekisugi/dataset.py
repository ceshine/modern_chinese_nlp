import numpy as np
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, x, y, backwards=False, sos=None, eos=None):
        self.x, self.y, self.backwards, self.sos, self.eos = x, y, backwards, sos, eos

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.backwards:
            x = list(reversed(x))
        if self.eos is not None:
            x = x + [self.eos]
        if self.sos is not None:
            x = [self.sos]+x
        return np.array(x), self.y[idx]

    def __len__(self):
        return len(self.x)
