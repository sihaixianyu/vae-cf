import os
import pickle

from process import process


class Dataset:
    def __init__(self, data_dir, data_name, leave_k=5, min_item_per_user=10, min_user_per_item=1, separator=',',
                 order_by_popularity=True):
        self.data_dir = data_dir
        self.leave_k = leave_k

        file_path_prefix = os.path.join(data_dir, data_name, data_name)
        rating_file_path = file_path_prefix + '.rating'
        data_file_path = file_path_prefix + '.data'
        info_file_path = file_path_prefix + '.info'
        if not (os.path.exists(data_file_path) and os.path.exists(info_file_path)):
            process(rating_file_path, leave_k, min_item_per_user, min_user_per_item, separator, order_by_popularity)

        with open(data_file_path, 'rb') as f:
            data_dict = pickle.load(f)

        with open(info_file_path, 'rb') as f:
            info_dict = pickle.load(f)

        self.train_sp_matrix = data_dict['train_sp_mat']
        # self.test_sp_matrix = data_dict['test_sp_mat']

        self.train_matrix = self.train_sp_matrix.toarray()
        # self.test_matrix = self.test_sp_matrix.toarray()

        # self.train_dict = data_dict['train_dict']
        self.test_dict = data_dict['test_dict']

        self.uid_map = info_dict['uid_map']
        self.iid_map = info_dict['iid_map']

        self.user_to_item_num = info_dict['user_to_item_num']
        self.item_to_user_num = info_dict['item_to_user_num']

        self.user_num = len(self.uid_map)
        self.item_num = len(self.iid_map)
