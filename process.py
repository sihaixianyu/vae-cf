import os
import pickle

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm


# 来自LOCA的数据处理方法
def process(file_path, leave_k=5, min_item_per_user=10, min_user_per_item=1, separator=',', order_by_popularity=True):
    print('Preprocess start!')
    raw_dict, uid_map, user_to_item_num, iid_map, item_to_user_num = read_raw_data(file_path, separator,
                                                                                   order_by_popularity)
    if min_item_per_user > 0:
        raw_dict, uid_map = filter_min_item_cnt(raw_dict, min_item_per_user, uid_map)

    train_sp_mat, test_sp_mat, train_dict, test_dict = leave_k_out(raw_dict, uid_map, iid_map, leave_k)

    data_to_save = {
        'train_sp_mat': train_sp_mat,
        'test_sp_mat': test_sp_mat,
        'train_dict': train_dict,
        'test_dict': test_dict,
    }

    info_to_save = {
        'uid_map': uid_map,
        'iid_map': iid_map,
        'user_to_item_num': user_to_item_num,
        'item_to_user_num': item_to_user_num,
    }

    path_prefix = os.path.splitext(file_path)[0]
    with open(path_prefix + '.data', 'wb') as f:
        pickle.dump(data_to_save, f)

    with open(path_prefix + '.info', 'wb') as f:
        pickle.dump(info_to_save, f)

    print('Preprocess end!')


# 读取原始数据，并且按照user和item的活跃度进行排序，映射原始id->排序后的id
def read_raw_data(file_path: str, separator=',', order_by_popularity=True):
    with open(file_path, "r") as f:
        lines = f.readlines()

    user_num, item_num = 0, 0
    user_to_item_num, item_to_user_num = {}, {}
    # 旧ID到新ID
    uid_map, iid_map = {}, {}

    for line in lines:
        uid, iid, _, _ = line.strip().split(separator)
        uid = int(uid)
        iid = int(iid)

        if uid not in uid_map:
            uid_map[uid] = user_num
            new_uid = user_num
            user_to_item_num[new_uid] = 1
            user_num += 1
        else:
            new_uid = uid_map[uid]
            user_to_item_num[new_uid] += 1

        if iid not in iid_map:
            iid_map[iid] = item_num
            new_iid = item_num
            item_to_user_num[new_iid] = 1
            item_num += 1
        else:
            new_iid = iid_map[iid]
            item_to_user_num[new_iid] += 1

    # 根据
    if order_by_popularity:
        uid_map, user_to_item_num = order_id_by_popularity(uid_map, user_to_item_num)
        iid_map, item_to_user_num = order_id_by_popularity(iid_map, item_to_user_num)

    raw_dict = {u: [] for u in user_to_item_num}
    for line in lines:
        uid, iid, rating, timestamp = line.strip().split(separator)
        uid = int(uid)
        iid = int(iid)
        rating = float(rating)
        timestamp = int(timestamp)
        raw_dict[uid_map[uid]].append((iid_map[iid], rating, timestamp))

    return raw_dict, uid_map, user_to_item_num, iid_map, item_to_user_num


def order_id_by_popularity(id_map: dict, id_to_num: dict):
    old_to_pop_map = {}
    new_to_pop_map = {}
    new_id_to_num = {}
    sorted_old_id_map = sorted(id_to_num.items(), key=lambda x: x[-1], reverse=True)
    for pop, new_pop_tuple in enumerate(sorted_old_id_map):
        new = new_pop_tuple[0]
        new_to_pop_map[new] = pop
        new_id_to_num[pop] = new_pop_tuple[1]
    for old, new in id_map.items():
        old_to_pop_map[old] = new_to_pop_map[new]

    return old_to_pop_map, new_id_to_num


def filter_min_item_cnt(raw_dict: dict, min_item_cnt: int, uid_map: dict) -> (dict, dict):
    sorted_uid_map = sorted(uid_map.items(), key=lambda x: x[-1], reverse=True)
    for old_uid, new_uid in sorted_uid_map:
        item_infos = raw_dict[new_uid]
        item_num = len(item_infos)

        # 计算剔除user的数量，方便后面进行重排序
        modifier = 0
        if item_num < min_item_cnt:
            raw_dict.pop(new_uid)
            uid_map.pop(old_uid)
            modifier += 1
        elif modifier > 0:
            item_infos[new_uid - modifier] = item_infos
            uid_map[old_uid] = new_uid - modifier
    return raw_dict, uid_map


def leave_k_out(raw_dict, uid_map, iid_map, leave_k):
    user_num = len(uid_map)
    item_num = len(iid_map)
    rating_num = 0

    train_dict = {u: [] for u in range(user_num)}
    train_mat = np.zeros((user_num, item_num))

    test_dict = {u: [] for u in range(user_num)}
    test_mat = np.zeros((user_num, item_num))

    for uid in tqdm(raw_dict):
        # 按照时间戳升序排序
        item_infos = sorted(raw_dict[uid], key=lambda x: x[-1])
        rating_num += len(item_infos)

        # 分离测试数据
        for i in range(leave_k):
            item_info = item_infos.pop()
            test_mat[uid, item_info[0]] = 1
            test_dict[uid].append(item_info[0])

        # 分离训练数据
        for item_info in item_infos:
            train_mat[uid, item_info[0]] = 1
            train_dict[uid].append(item_info[0])

    train_sp_mat = csr_matrix(train_mat, shape=(user_num, item_num))
    test_sp_mat = csr_matrix(test_mat, shape=(user_num, item_num))

    return train_sp_mat, test_sp_mat, train_dict, test_dict
