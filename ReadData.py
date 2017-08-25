import numpy as np
from os.path import exists
import UserFeatures
import random
from scipy.sparse import csc_matrix
import ItemFeatures


def read_rating(path, num_users, num_items, num_u_features, num_total_ratings, a, b, train_ratio,random_seed):
    fp = open(path + "ratings.dat")

    user_train_set = set()
    user_test_set = set()
    item_train_set = set()
    item_test_set = set()

    user_features = UserFeatures.getUserFeature()

    R = np.zeros((num_users,num_items ))
    mask_R = np.zeros((num_users, num_items))

    train_R = np.zeros((num_users, num_items  + num_u_features))
    train_R_4_test = np.zeros((num_users, num_items  + num_u_features))
    test_R = np.zeros((num_users, num_items ))

    train_mask_R = np.zeros((num_users, num_items + num_u_features))
    train_mask_R_4_test = np.zeros((num_users, num_items  + num_u_features))
    test_mask_R = np.zeros((num_users, num_items ))

    random_perm_idx = np.random.permutation(num_total_ratings)
    train_idx = random_perm_idx[0:int(num_total_ratings*train_ratio)]
    test_idx = random_perm_idx[int(num_total_ratings*train_ratio):]

    num_train_ratings = len(train_idx)
    num_test_ratings = len(test_idx)



    lines = fp.readlines()
    for line in lines:
        user,item,rating,_ = line.split("::")
        user_idx = int(user) - 1
        item_idx = int(item) - 1
        R[user_idx,item_idx] = int(rating)
        mask_R[user_idx,item_idx] = 1

    ''' Train '''
    for itr in train_idx:
        line = lines[itr]
        user,item,rating,_ = line.split("::")
        user_idx = int(user) - 1
        item_idx = int(item) - 1

        if int(rating) >= 4:
            train_R[user_idx, item_idx ] = int(rating)
            train_R_4_test[user_idx,  item_idx ] = int(rating)

        train_mask_R[user_idx, item_idx] = 1
        train_mask_R_4_test[user_idx, item_idx ] = 1

        user_train_set.add(user_idx)
        item_train_set.add(item_idx)

    for u in user_train_set:
        #print(u)
        train_R[u, num_items: num_items + num_u_features] = np.array(user_features[u + 1]) * 5
        train_mask_R[u, num_items: num_items  + num_u_features] = [1] * num_u_features
        train_R_4_test[u, num_items : num_items  + num_u_features] = np.array(user_features[u + 1]) * 5
        train_mask_R_4_test[u, num_items : num_items  + num_u_features] = [1] * num_u_features

    ''' Test '''
    for itr in test_idx:
        line = lines[itr]
        user, item, rating, _ = line.split("::")
        user_idx = int(user) - 1
        item_idx = int(item) - 1
        if int(rating) >= 4:
            test_R[user_idx,  item_idx] = 1

        test_mask_R[user_idx,  item_idx] = 1

        user_test_set.add(user_idx)
        item_test_set.add(item_idx)

    for u in user_test_set:
        train_R_4_test[u, num_items : num_items  + num_u_features] = np.array(user_features[u + 1]) * 5
        train_mask_R_4_test[u, num_items : num_items  + num_u_features] = [1] * num_u_features

    return R, mask_R, train_R, train_mask_R, test_R, test_mask_R,num_train_ratings,num_test_ratings,\
user_train_set,item_train_set,user_test_set,item_test_set, train_R_4_test, train_mask_R_4_test

def read_rating_rating(path, num_users, num_items, num_i_features, num_total_ratings, train_ratio):
    fp = open("./data/ml-100k/u.data")
    #fp = open(path + "ratings.dat")

    user_train_set = set()
    user_test_set = set()
    item_train_set = set()
    item_test_set = set()
    item_features = ItemFeatures.getItemFeatures100k()

    feature_weight = 1
    R = np.zeros((num_users,num_items ))
    mask_R = np.zeros((num_users, num_items))
    train_R = np.zeros((num_users + num_i_features, num_items))
    test_R = np.zeros((num_users, num_items))
    train_R_4_test = np.zeros((num_users + num_i_features, num_items))

    train_mask_R = np.zeros((num_users + num_i_features, num_items))
    train_mask_R_4_test = np.zeros((num_users + num_i_features, num_items))
    test_mask_R = np.zeros((num_users, num_items))

    random_perm_idx = np.random.permutation(num_total_ratings)
    train_idx = random_perm_idx[0:int(num_total_ratings * train_ratio)]
    test_idx = random_perm_idx[int(num_total_ratings * train_ratio):]

    num_train_ratings = len(train_idx)
    num_test_ratings = len(test_idx)

    lines = fp.readlines()

    ''' Train '''
    for itr in train_idx:
        line = lines[itr]
        user, item, rating, _ = line.split("\t")
        user_idx = int(user) - 1
        item_idx = int(item) - 1
        train_R[user_idx, item_idx] = int(rating)
        train_mask_R[user_idx, item_idx] = 1
        train_R_4_test[user_idx, item_idx] = int(rating)
        train_mask_R_4_test[user_idx, item_idx] = 1
        user_train_set.add(user_idx)
        item_train_set.add(item_idx)

    for i in item_train_set:
        train_R[num_users: num_users + num_i_features, i] = np.array(item_features[i]) * feature_weight
        train_mask_R[num_users: num_users + num_i_features, i] = [1] * num_i_features
        train_R_4_test[num_users: num_users + num_i_features, i] = np.array(item_features[i]) * feature_weight
        train_mask_R_4_test[num_users: num_users + num_i_features, i] = [1] * num_i_features

    ''' Test '''
    for itr in test_idx:
        line = lines[itr]
        user, item, rating, _ = line.split("\t")
        user_idx = int(user) - 1
        item_idx = int(item) - 1
        test_R[user_idx, item_idx] = int(rating)
        test_mask_R[user_idx, item_idx] = 1

        user_test_set.add(user_idx)
        item_test_set.add(item_idx)

    for i in item_test_set:
        train_R_4_test[num_users: num_users + num_i_features, i] = np.array(item_features[i]) * feature_weight
        train_mask_R_4_test[num_users: num_users + num_i_features, i] = [1] * num_i_features

    return R, mask_R, train_R, train_mask_R, test_R, test_mask_R,num_train_ratings,num_test_ratings,\
user_train_set,item_train_set,user_test_set,item_test_set, train_R_4_test, train_mask_R_4_test




def read_rating_ranking(path, num_users, num_items, num_u_features, num_total_ratings, train_ratio):
    fp = open("./data/ml-100k/u.data")

    user_train_set = set()
    user_test_set = set()
    item_train_set = set()
    item_test_set = set()

    user_features = UserFeatures.getUserFeatures100K()

    R = np.zeros((num_users,num_items ))
    mask_R = np.zeros((num_users, num_items))

    train_R = np.zeros((num_users, num_items * 1 + num_u_features))
    test_R = np.zeros((num_users, num_items ))

    train_mask_R = np.zeros((num_users, num_items * 1 + num_u_features))
    test_mask_R = np.zeros((num_users, num_items ))

    train_R_4_test = np.zeros((num_users, num_items * 1 + num_u_features))
    train_mask_R_4_test = np.zeros((num_users, num_items * 1 + num_u_features))

    random_perm_idx = np.random.permutation(num_total_ratings)
    train_idx = random_perm_idx[0:int(num_total_ratings*train_ratio)]
    test_idx = random_perm_idx[int(num_total_ratings*train_ratio):]

    num_train_ratings = len(train_idx)
    num_test_ratings = len(test_idx)

    lines = fp.readlines()
    for line in lines:
        user,item,rating,_ = line.split("\t")
        user_idx = int(user) - 1
        item_idx = int(item) - 1
        R[user_idx,item_idx] = int(rating)
        mask_R[user_idx,item_idx] = 1

    ''' Train '''
    for itr in train_idx:
        line = lines[itr]
        user,item,rating,_ = line.split("\t")
        user_idx = int(user) - 1
        item_idx = int(item) - 1
        if int(rating) > 4:
            train_R[user_idx,  item_idx] = 1
            train_mask_R[user_idx,  item_idx] = 1
            train_R_4_test[user_idx,  item_idx] = 1
            train_mask_R_4_test[user_idx,  item_idx] = 1

        user_train_set.add(user_idx)
        item_train_set.add(item_idx)


    for u in user_train_set:
        # print(u)
        train_R[u, num_items : num_items  + num_u_features] = np.array(user_features[u + 1])
        train_mask_R[u, num_items : num_items  + num_u_features] = [1] * num_u_features
        train_R_4_test[u, num_items : num_items  + num_u_features] = np.array(user_features[u + 1])
        train_mask_R_4_test[u, num_items : num_items  + num_u_features] = [1] * num_u_features

    ''' Test '''
    for itr in test_idx:
        line = lines[itr]
        user, item, rating, _ = line.split("\t")
        user_idx = int(user) - 1
        item_idx = int(item) - 1

        if int(rating) > 4:
            test_R[user_idx,  item_idx] = 1
            test_mask_R[user_idx,  item_idx] = 1

        user_test_set.add(user_idx)
        item_test_set.add(item_idx)

    for u in user_test_set:
        train_R_4_test[u, num_items : num_items  + num_u_features] = np.array(user_features[u + 1])
        train_mask_R_4_test[u, num_items : num_items  + num_u_features] = [1] * num_u_features

    return R, mask_R, train_R, train_mask_R, test_R, test_mask_R,num_train_ratings,num_test_ratings,\
user_train_set,item_train_set,user_test_set,item_test_set,train_R_4_test,train_mask_R_4_test




