import time

import torch
import torch.optim as optim

from batcher import Batcher
from dataset import Dataset
from evaluator import Evaluator
from model import VAE
from trainer import Trainer
from util import save_user_embdding, print_res

if __name__ == '__main__':
    data_dir = 'data/'
    data_name = 'ml-1m'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Dataset('data/', 'ml-1m', separator='::')

    train_batcher = Batcher(dataset, batch_size=512, shuffle=True)
    test_batcher = Batcher(dataset, batch_size=1024, shuffle=True)

    enc_dec_dims = [dataset.item_num]
    enc_dec_dims.extend([300])
    model = VAE(enc_dec_dims, enc_dec_dims[::-1], device=device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)

    trainer = Trainer(train_batcher, model, optimizer)
    evaluator = Evaluator(test_batcher, model, top_k=10)

    best_epoch = {
        'epoch': 0,
        'prec': .0,
        'recall': .0,
        'ndcg': .0,
    }
    for epoch in range(1, 101):

        train_start = time.time()
        loss = trainer.train()
        train_time = time.time() - train_start

        eval_start = time.time()
        prec, recall, ndcg = evaluator.evaluate()
        eval_time = time.time() - eval_start

        print('Epoch=%3d, Loss=%3d, Prec=%.4f, Recall=%.4f, NDCG=%.4f, Time=(%.4f + %.4f)'
              % (epoch, loss, prec, recall, ndcg, train_time, eval_time))

        if best_epoch['prec'] <= prec:
            best_epoch['epoch'] = epoch
            best_epoch['prec'] = prec
            best_epoch['recall'] = recall
            best_epoch['ndcg'] = ndcg

            if epoch >= 20:
                train_tensor = torch.FloatTensor(dataset.train_matrix)
                user_embedding = model.calc_user_embedding(train_tensor, test_batcher)
                save_user_embdding(data_dir, data_name, user_embedding)

    print_res('Best Epoch: %3d, Prec: %.4f, Recall: %.4f, NDCG: %.4f'
              % (best_epoch['epoch'], best_epoch['prec'], best_epoch['recall'], best_epoch['ndcg']))
