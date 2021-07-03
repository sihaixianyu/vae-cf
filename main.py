import time

import torch
import torch.optim as optim

from batcher import BaseBatcher
from dataset import BaseDataset
from evaluator import Evaluator
from model import VAE
from trainer import Trainer
from util import res_print

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = BaseDataset('data/', 'agame', separator=',')

    train_batcher = BaseBatcher(dataset, batch_size=512, shuffle=True)
    test_batcher = BaseBatcher(dataset, batch_size=1024, shuffle=True)

    enc_dec_dims = [dataset.item_num]
    enc_dec_dims.extend([300])
    model = VAE(enc_dec_dims, enc_dec_dims[::-1], device=device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)

    trainer = Trainer(train_batcher, model, optimizer)
    evaluator = Evaluator(test_batcher, model, top_k=50)

    best_epoch = {
        'epoch': 0,
        'prec': .0,
        'recall': .0,
        'ndcg': .0,
    }
    for epoch in range(1, 201):

        train_start = time.time()
        loss = trainer.train()
        train_time = time.time() - train_start

        eval_start = time.time()
        prec, recall, ndcg = evaluator.evaluate()
        eval_time = time.time() - eval_start

        print('Epoch=%3d, Loss=%3d, Prec=%.4f, Recall=%.4f, NDCG=%.4f, Time=(%.4f + %.4f)'
              % epoch, loss, prec, recall, ndcg, train_time, eval_time)

        if best_epoch['prec'] <= prec:
            best_epoch['epoch'] = epoch
            best_epoch['prec'] = prec
            best_epoch['recall'] = recall
            best_epoch['ndcg'] = ndcg

    res_print('Best Epoch: %3d, Prec: %.4f, Recall: %.4f, NDCG: %.4f'
              % (best_epoch['epoch'], best_epoch['prec'], best_epoch['recall'], best_epoch['ndcg']))
