import math
import time
from collections import OrderedDict

import numpy as np
import torch

from batcher import BaseBatcher
from util import Statistic
from model import VAE


class Evaluator:
    def __init__(self, batcher: BaseBatcher, model: VAE, top_k=20):
        self.batcher = batcher
        self.model = model
        self.top_k = top_k

        self.batch_size = batcher.batch_size
        self.leave_k = batcher.dataset.leave_k
        self.train_matrix = batcher.dataset.train_matrix
        self.test_dict = batcher.dataset.test_dict

        self.prec_stat = Statistic('{}@{}'.format('Prec', self.top_k))
        self.recall_stat = Statistic('{}@{}'.format('Recall', self.top_k))
        self.ndcg_stat = Statistic('{}@{}'.format('NDCG', self.top_k))

    def evaluate(self) -> float:
        start = time.time()
        self.model.eval()

        for uids in self.batcher:
            batch_matrix = self.train_matrix[uids]
            batch_tensor = torch.FloatTensor(batch_matrix).to(self.model.device)
            eval_dict = {uid: self.test_dict[uid] for uid in uids}

            with torch.no_grad():
                pred_tensor = self.model.forward(batch_tensor)
                pred_matrix: np.ndarray = pred_tensor.detach().cpu().numpy()
                # 排除掉已经交互过的项目
                pred_matrix[np.nonzero(batch_matrix)] = float('-inf')

            top_matrix = self.predict_top_mat(pred_matrix.astype(np.float32))
            self.calc_matric(top_matrix.astype(np.int64), eval_dict)

        end = time.time()
        return end - start

    def predict_top_mat(self, pred_matrix: np.ndarray) -> np.ndarray:
        # 根据第k大的元素划分数据，前面的元素都大于它，后面的都小于它，返回划分后的数据索引
        top_item_idxs = np.argpartition(-pred_matrix, self.top_k, 1)[:, 0:self.top_k]
        # 此时的前top_k个元素不保证排序，我们通过原始索引获得原始值
        top_item_vals = np.take_along_axis(pred_matrix, top_item_idxs, 1)
        # 通过对top_k原始值进行排序,获得在它们排序后的top_k索引
        sorted_top_idxs = np.argsort(-top_item_vals, 1)
        # 通过原始值的top_k索引，获得排序后的原始索引
        sorted_item_idxs = np.take_along_axis(top_item_idxs, sorted_top_idxs, 1)

        return sorted_item_idxs

    def calc_matric(self, top_matrix: np.ndarray, eval_dict: dict):
        for i, uid in enumerate(eval_dict):
            pred_items = top_matrix[i]
            real_items = eval_dict[uid]

            hit_items = [(i + 1, item) for i, item in enumerate(pred_items) if item in real_items]

            idcg = 0.0
            for j in range(1, len(real_items) + 1):
                idcg += 1 / math.log((j + 1), 2)

            dcg = 0.0
            for k, item in hit_items:
                dcg += 1 / math.log(k + 1, 2)

            prec = len(hit_items) / self.top_k
            recall = len(hit_items) / len(real_items)
            ndcg = dcg / idcg

            self.prec_stat.update(prec)
            self.recall_stat.update(recall)
            self.ndcg_stat.update(ndcg)
