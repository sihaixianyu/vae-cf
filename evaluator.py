import math

import numpy as np
import torch

from batcher import BaseBatcher
from model import VAE


class Evaluator:
    def __init__(self, batcher: BaseBatcher, model: VAE, top_k=20):
        self.batcher = batcher
        self.model = model
        self.top_k = top_k

        self.train_matrix = batcher.dataset.train_matrix
        self.test_dict = batcher.dataset.test_dict

    def evaluate(self) -> (float, float, float):
        self.model.eval()

        total_prec, total_recall, total_ndcg = .0, .0, .0
        for uids in self.batcher:
            batch_matrix = self.train_matrix[uids]
            batch_tensor = torch.FloatTensor(batch_matrix).to(self.model.device)
            eval_dict = {uid: self.test_dict[uid] for uid in uids}

            with torch.no_grad():
                pred_tensor = self.model.forward(batch_tensor)
                pred_matrix: np.ndarray = pred_tensor.detach().cpu().numpy()
                # 排除掉已经交互过的项目
                pred_matrix[np.nonzero(batch_matrix)] = float('-inf')

            top_matrix = self.predict_top(pred_matrix.astype(np.float64)).astype(np.int64)
            batch_prec, batch_recall, batch_ndcg = self.calc_matric(top_matrix, eval_dict)

            total_prec += batch_prec
            total_recall += batch_recall
            total_ndcg += batch_ndcg

        prec = total_prec / len(self.batcher)
        recall = total_recall / len(self.batcher)
        ndcg = total_ndcg / len(self.batcher)

        return prec, recall, ndcg

    def predict_top(self, pred_matrix: np.ndarray) -> np.ndarray:
        # 根据第k大的元素划分数据，前面的元素都大于它，后面的都小于它，返回划分后的数据索引
        top_item_idxs = np.argpartition(-pred_matrix, self.top_k, 1)[:, 0:self.top_k]
        # 此时的前top_k个元素不保证排序，我们通过原始索引获得原始值
        top_item_vals = np.take_along_axis(pred_matrix, top_item_idxs, 1)
        # 通过对top_k原始值进行排序,获得在它们排序后的top_k索引
        sorted_top_idxs = np.argsort(-top_item_vals, 1)
        # 通过原始值的top_k索引，获得排序后的原始索引
        sorted_item_idxs = np.take_along_axis(top_item_idxs, sorted_top_idxs, 1)

        return sorted_item_idxs

    def calc_matric(self, top_matrix: np.ndarray, eval_dict: dict) -> (float, float, float):
        total_prec, total_recall, total_ndcg = .0, .0, .0
        for i, uid in enumerate(eval_dict):
            pred_items = top_matrix[i]
            real_items = eval_dict[uid]

            hit_items = [(i + 1, item) for i, item in enumerate(pred_items) if item in real_items]

            batch_idcg = 0.0
            for j in range(1, len(real_items) + 1):
                batch_idcg += 1 / math.log((j + 1), 2)

            batch_dcg = 0.0
            for k, item in hit_items:
                batch_dcg += 1 / math.log(k + 1, 2)

            total_prec += len(hit_items) / self.top_k
            total_recall += len(hit_items) / len(real_items)
            total_ndcg += batch_dcg / batch_idcg

        eval_num = len(eval_dict)
        batch_prec = total_prec / eval_num
        batch_recall = total_recall / eval_num
        batch_ndcg = total_ndcg / eval_num

        return batch_prec, batch_recall, batch_ndcg
