import numpy as np

from dataset import BaseDataset


class BaseBatcher:
    def __init__(self, dataset: BaseDataset, batch_size=256, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return (self.dataset.user_num // self.batch_size) + 1

    def __iter__(self):
        shuffled_uids = np.random.permutation(self.dataset.user_num)

        cnt = 0
        uids = []
        for uid in shuffled_uids:
            if cnt == self.batch_size:
                yield uids
                uids = []
                cnt = 0
            else:
                uids.append(uid)
                cnt += 1
        yield uids
