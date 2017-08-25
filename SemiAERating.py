import tensorflow as tf
import time
import numpy as np
import os

class SemiAERating():
    def __init__(self,sess,
                 num_users,num_items, num_i_features, hidden_neuron,f_act,g_act,
                  train_R, train_mask_R, test_R, test_mask_R,train_R_4_test, train_mask_R_4_test, num_train_ratings,num_test_ratings,
                 train_epoch,batch_size,lr,optimizer_method,
                 display_step,random_seed,
                 decay_epoch_step,lambda_value,
                 user_train_set, item_train_set, user_test_set, item_test_set):

        self.sess = sess
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_neuron = hidden_neuron
        self.num_i_features = num_i_features

        self.train_R = train_R
        self.train_mask_R = train_mask_R
        self.train_R_4_test = train_R_4_test
        self.train_mask_R_4_test = train_mask_R_4_test
        self.test_R = test_R
        self.test_mask_R = test_mask_R
        self.num_train_ratings = num_train_ratings
        self.num_test_ratings = num_test_ratings

        self.train_epoch = train_epoch
        self.batch_size = batch_size
        self.num_batch = int(self.num_items / float(self.batch_size)) + 1

        self.lr = lr
        self.optimizer_method = optimizer_method
        self.display_step = display_step
        self.random_seed = random_seed

        self.f_act = f_act
        self.g_act = g_act

        self.global_step = tf.Variable(0, trainable=False)
        self.decay_epoch_step = decay_epoch_step
        self.decay_step = self.decay_epoch_step * self.num_batch

        self.lambda_value = lambda_value

        self.train_cost_list = []
        self.test_cost_list = []
        self.test_rmse_list = []

        self.user_train_set = user_train_set
        self.item_train_set = item_train_set
        self.user_test_set = user_test_set
        self.item_test_set = item_test_set


    def execute(self):
        self.prepare_model()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch_itr in range(self.train_epoch):
            self.train_model(epoch_itr)
            self.test_model(epoch_itr)

    def prepare_model(self):
        self.input_R = tf.placeholder(dtype=tf.float32, shape=[self.num_users + self.num_i_features, None])
        self.input_mask_R = tf.placeholder(dtype=tf.float32, shape=[self.num_users + self.num_i_features, None])

        V = tf.get_variable(name="V", initializer=tf.truncated_normal(
            shape=[self.hidden_neuron, self.num_users + self.num_i_features],
            mean=0, stddev=0.03), dtype=tf.float32)
        W = tf.get_variable(name="W", initializer=tf.truncated_normal(shape=[self.num_users, self.hidden_neuron],
                                                                      mean=0, stddev=0.03), dtype=tf.float32)

        mu = tf.get_variable(name="mu", initializer=tf.zeros(shape=self.hidden_neuron), dtype=tf.float32)
        b = tf.get_variable(name="b", initializer=tf.zeros(shape=self.num_users), dtype=tf.float32)

        pre_Encoder = tf.add(tf.expand_dims(mu, 1), tf.matmul(V, self.input_R))

        self.Encoder = self.g_act(pre_Encoder)
        pre_Decoder = tf.matmul(W, self.Encoder) + tf.expand_dims(b, 1)

        self.Decoder = self.f_act(pre_Decoder)

        pre_cost1 = tf.multiply((self.input_R[0:self.num_users, :] - self.Decoder),
                                self.input_mask_R[0:self.num_users, :])
        cost1 = tf.square(tf.norm(pre_cost1))
        pre_cost2 = tf.square(tf.norm(W)) + tf.square(tf.norm(V))
        cost2 = self.lambda_value * 0.5 * pre_cost2

        self.cost = cost1 + cost2

        if self.optimizer_method == "Adam":
            optimizer = tf.train.AdamOptimizer(self.lr)
        elif self.optimizer_method == "Adadelta":
            optimizer = tf.train.AdadeltaOptimizer(self.lr)
        elif self.optimizer_method == "Adagrad":
            optimizer = tf.train.AdadeltaOptimizer(self.lr)
        elif self.optimizer_method == "RMSProp":
            optimizer = tf.train.RMSPropOptimizer(self.lr)
        elif self.optimizer_method == "GradientDescent":
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif self.optimizer_method == "Momentum":
            optimizer = tf.train.MomentumOptimizer(self.lr, 0.9)
        else:
            raise ValueError("Optimizer Key ERROR")

        gvs = optimizer.compute_gradients(self.cost)
        capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
        self.optimizer = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)

    def train_model(self, itr):
        random_perm_doc_idx = np.random.permutation(self.num_items)

        batch_cost = 0
        for i in range(self.num_batch):
            if i == self.num_batch - 1:
                batch_set_idx = random_perm_doc_idx[i * self.batch_size:]
            elif i < self.num_batch - 1:
                batch_set_idx = random_perm_doc_idx[i * self.batch_size: (i + 1) * self.batch_size]

            _, Cost = self.sess.run([self.optimizer, self.cost],
                                    feed_dict={self.input_R: self.train_R[:, batch_set_idx],
                                               self.input_mask_R: self.train_mask_R[:, batch_set_idx]})

            batch_cost = batch_cost + Cost
        self.train_cost_list.append(batch_cost)

        # if itr % self.display_step == 0:
        print("Training...", "Epoch %d..." % (itr), " Total cost = {:.2f}".format(batch_cost))

    def test_model(self, itr):
        start_time = time.time()
        Cost, Decoder = self.sess.run(
            [self.cost, self.Decoder],
            feed_dict={self.input_R: self.train_R_4_test,
                       self.input_mask_R: self.train_mask_R_4_test})

        self.test_cost_list.append(Cost)

        if itr % self.display_step == 0:
            Estimated_R = Decoder.clip(min=1, max=5)
            unseen_user_test_list = list(self.user_test_set - self.user_train_set)
            unseen_item_test_list = list(self.item_test_set - self.item_train_set)

            for user in unseen_user_test_list:
                for item in unseen_item_test_list:
                    if self.test_mask_R[item, user] == 1:
                        Estimated_R[item, user] = 3

            pre_numerator = np.multiply((Estimated_R - self.test_R), self.test_mask_R)
            numerator = np.sum(np.square(pre_numerator))
            denominator = self.num_test_ratings
            RMSE = np.sqrt(numerator / float(denominator))
            MAE = np.sum(np.abs(pre_numerator)) /self.num_test_ratings

            self.test_rmse_list.append(RMSE)

            print("Testing...", "Epoch %d ..." % (itr), " Total cost = {:.2f}".format(Cost),
                  " RMSE = {:.5f}".format(RMSE),
                  " MAE = {:.5f}".format(MAE))



    def l2_norm(self, tensor):
        return tf.sqrt(tf.reduce_sum(tf.square(tensor)))



