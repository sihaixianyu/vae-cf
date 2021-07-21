import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer

from batcher import BaseBatcher
from vae import VAE


class Trainer:
    def __init__(self, batcher: BaseBatcher, model: VAE, optimizer: Optimizer):
        self.batcher = batcher
        self.model = model
        self.optimizer = optimizer

        self.train_matrix = batcher.dataset.train_matrix

    def train(self) -> float:
        self.model.train()

        total_loss = .0
        for uids in self.batcher:
            batch_matrix = self.train_matrix[uids]
            batch_tensor = torch.FloatTensor(batch_matrix).to(self.model.device)

            if self.model.anneal_num > 0:
                self.model.anneal = min(self.model.anneal_cap, 1. * self.model.update_cnt / self.model.anneal_num)
            else:
                self.model.anneal = self.model.anneal_cap

            batch_loss = self.train_per_batch(batch_tensor)
            total_loss += batch_loss

        loss = total_loss / len(self.batcher)

        return loss

    def train_per_batch(self, batch_tensor: torch.Tensor) -> float:
        self.optimizer.zero_grad()

        y, kl_loss = self.model.forward(batch_tensor)
        ce_loss = -(F.log_softmax(y, 1) * batch_tensor).sum(1).mean()

        batch_loss: torch.Tensor = ce_loss + kl_loss * self.model.anneal
        batch_loss.backward()

        self.model.update_cnt += 1

        self.optimizer.step()
        return batch_loss.detach().cpu().numpy().astype(np.float64)
