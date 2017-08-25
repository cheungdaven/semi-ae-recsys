from ReadData import *
from SemiAERanking import SemiAERanking
from SemiAERating import SemiAERating
import tensorflow as tf


def RatingPrediction():
    train_ratio = 0.8
    hidden_neuron = 500  # 1000
    random_seed = 1
    batch_size = 500  # 1000 #256 #1024
    lr = 1e-3  # learning rate
    train_epoch = 50  #
    optimizer_method = 'Adam'
    display_step = 1
    decay_epoch_step = 0
    lambda_value = 1
    f_act = tf.identity
    g_act = tf.nn.sigmoid
    num_users = 943
    num_items = 1682
    num_u_features = 39  # 30
    num_total_ratings = 100000
    path = "./data/ml-100k/"

    R, mask_R, train_R, train_mask_R, test_R, test_mask_R,num_train_ratings,num_test_ratings,\
    user_train_set,item_train_set,user_test_set,item_test_set, train_R_4_test, train_mask_R_4_test \
        = read_rating_rating(path, num_users, num_items, num_u_features, num_total_ratings, train_ratio)


    with tf.Session() as sess:
        SARating = SemiAERating(sess,num_users,num_items, num_u_features, hidden_neuron,f_act,g_act,
                             train_R, train_mask_R, test_R, test_mask_R, train_R_4_test, train_mask_R_4_test,num_train_ratings,num_test_ratings,
                             train_epoch,batch_size, lr, optimizer_method, display_step, random_seed,
                             decay_epoch_step,lambda_value, user_train_set, item_train_set, user_test_set, item_test_set)

        SARating.execute()

def RankingPrediction():
    train_ratio = 0.5
    hidden_neuron = 10  # 1000
    random_seed = 1
    batch_size = 500  # 1000 #256 #1024
    lr = 1e-3  # learning rate
    train_epoch = 800  #
    optimizer_method = 'Adam'
    display_step = 1
    lambda_value = 1
    f_act = tf.identity
    g_act = tf.nn.sigmoid
    num_users = 943
    num_items = 1682
    num_u_features = 30  # 39
    num_total_ratings = 100000
    path = "./data/ml-100k/"

    R, mask_R, train_R, train_mask_R, test_R, test_mask_R, num_train_ratings, num_test_ratings, \
    user_train_set, item_train_set, user_test_set, item_test_set, train_R_4_test, train_mask_R_4_test \
        = read_rating_ranking(path, num_users, num_items, num_u_features, num_total_ratings, train_ratio)

    with tf.Session() as sess:
        SARanking = SemiAERanking(sess, num_users, num_items, num_u_features, hidden_neuron, f_act, g_act,
                          train_R, train_mask_R, test_R, test_mask_R, train_R_4_test, train_mask_R_4_test, num_train_ratings, num_test_ratings,
                          train_epoch, batch_size, lr, optimizer_method, display_step, random_seed, lambda_value,
                          user_train_set, item_train_set, user_test_set, item_test_set)

        SARanking.execute()


if __name__ == '__main__':
    RatingPrediction()