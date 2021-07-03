import time

import torch
import torch.nn.functional as F
from torch.optim import Optimizer

from batcher import BaseBatcher
from model import VAE
from util import Statistic


class Trainer:
    def __init__(self, batcher: BaseBatcher, model: VAE, optimizer: Optimizer):
        self.batcher = batcher
        self.model = model
        self.optimizer = optimizer

        self.train_matrix = batcher.dataset.train_matrix

    def train(self) -> float:
        self.model.train()

        for uids in self.batcher:
            batch_matrix = self.train_matrix[uids]
            batch_tensor = torch.FloatTensor(batch_matrix).to(self.model.device)

            if self.model.anneal_num > 0:
                self.model.anneal = min(self.model.anneal_cap, 1. * self.model.update_cnt / self.model.anneal_num)
            else:
                self.model.anneal = self.model.anneal_cap

            self.train_per_batch(batch_tensor)

        return loss

    def train_per_batch(self, batch_tensor: torch.Tensor):
        self.optimizer.zero_grad()

        y, kl_loss = self. model.forward(batch_tensor)
        ce_loss = -(F.log_softmax(y, 1) * batch_tensor).sum(1).mean()

        loss: torch.Tensor = ce_loss + kl_loss * self.model.anneal
        loss.backward()

        self.optimizer.step()

        self.model.update_cnt += 1

