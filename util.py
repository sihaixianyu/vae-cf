import os
import pickle

import numpy as np


def print_res(content: str):
    print('-' * 100)
    print(content)
    print('-' * 100)


def save_user_embdding(data_dir: str, data_name: str, user_embedding: np.ndarray):
    with open(os.path.join(data_dir, data_name, data_name) + '.emb', 'wb') as f:
        pickle.dump(user_embedding, f)


def load_user_embedding(data_dir: str, data_name: str) -> np.ndarray:
    with open(os.path.join(data_dir, data_name, data_name) + '.emb', 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    embedding = load_user_embedding('data/', 'ml-1m')
    print()
