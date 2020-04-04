# -*- Encoding:UTF-8 -*-

import tensorflow as tf
import numpy as np
import argparse
from DataSet import DataSet
import sys
import os
import heapq
import math
import json

def write_json(data,dst_path):
    with open(dst_path, 'w') as outfile:
        json.dump(data, outfile)

def main():
    parser = argparse.ArgumentParser(description="Options")

    parser.add_argument('-dataName', action='store', dest='dataName', default='ml-1m')
    parser.add_argument('-negNum', action='store', dest='negNum', default=7, type=int)
    parser.add_argument('-userLayer', action='store', dest='userLayer', default=[50, 64]) #default=[512, 64])
    parser.add_argument('-itemLayer', action='store', dest='itemLayer', default=[10, 64]) #default=[1024, 64])
    # parser.add_argument('-reg', action='store', dest='reg', default=1e-3)
    parser.add_argument('-lr', action='store', dest='lr', default=0.0001)
    parser.add_argument('-maxEpochs', action='store', dest='maxEpochs', default=50, type=int)#default=50, type=int)
    parser.add_argument('-batchSize', action='store', dest='batchSize', default=25, type=int)#default=256, type=int)
    parser.add_argument('-earlyStop', action='store', dest='earlyStop', default=5)
    parser.add_argument('-checkPoint', action='store', dest='checkPoint', default='./checkPoint/')
    parser.add_argument('-topK', action='store', dest='topK', default=10)

    args = parser.parse_args()

    classifier = Model(args)

    classifier.run()


class Model:
    def __init__(self, args):
        self.dataName = args.dataName
        self.dataSet = DataSet(self.dataName)
        self.shape = self.dataSet.shape
        self.maxRate = self.dataSet.maxRate

        self.train = self.dataSet.train
        self.test = self.dataSet.test

        self.negNum = args.negNum
        # self.testNeg = self.dataSet.getTestNeg(self.test, 99)
        self.testNeg = self.dataSet.getTestNeg(self.test, 31)
        self.add_embedding_matrix()

        self.add_placeholders()

        self.userLayer = args.userLayer
        self.itemLayer = args.itemLayer
        self.add_model()

        self.add_loss()

        self.lr = args.lr
        self.add_train_step()

        self.checkPoint = args.checkPoint
        self.init_sess()

        self.maxEpochs = args.maxEpochs
        self.batchSize = args.batchSize

        self.topK = args.topK
        self.earlyStop = args.earlyStop


    def add_placeholders(self):
        self.user = tf.placeholder(tf.int32)
        self.item = tf.placeholder(tf.int32)
        self.rate = tf.placeholder(tf.float32)
        self.drop = tf.placeholder(tf.float32)

    def add_embedding_matrix(self):
        self.user_item_embedding = tf.convert_to_tensor(self.dataSet.getEmbedding())
        self.item_user_embedding = tf.transpose(self.user_item_embedding)

    def add_model(self):
        user_input = tf.nn.embedding_lookup(self.user_item_embedding, self.user)
        item_input = tf.nn.embedding_lookup(self.item_user_embedding, self.item)

        def init_variable(shape, name):
            return tf.Variable(tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.01), name=name)

        with tf.name_scope("User_Layer"):
            user_W1 = init_variable([self.shape[1], self.userLayer[0]], "user_W1")
            user_out = tf.matmul(user_input, user_W1)
            for i in range(0, len(self.userLayer)-1):
                W = init_variable([self.userLayer[i], self.userLayer[i+1]], "user_W"+str(i+2))
                b = init_variable([self.userLayer[i+1]], "user_b"+str(i+2))
                user_out = tf.nn.relu(tf.add(tf.matmul(user_out, W), b))

        with tf.name_scope("Item_Layer"):
            item_W1 = init_variable([self.shape[0], self.itemLayer[0]], "item_W1")
            item_out = tf.matmul(item_input, item_W1)
            for i in range(0, len(self.itemLayer)-1):
                W = init_variable([self.itemLayer[i], self.itemLayer[i+1]], "item_W"+str(i+2))
                b = init_variable([self.itemLayer[i+1]], "item_b"+str(i+2))
                item_out = tf.nn.relu(tf.add(tf.matmul(item_out, W), b))

        norm_user_output = tf.sqrt(tf.reduce_sum(tf.square(user_out), axis=1))
        norm_item_output = tf.sqrt(tf.reduce_sum(tf.square(item_out), axis=1))
        self.y_ = tf.reduce_sum(tf.multiply(user_out, item_out), axis=1, keep_dims=False) / (norm_item_output* norm_user_output)
        self.y_ = tf.maximum(1e-6, self.y_)

    def add_loss(self):
        regRate = self.rate / self.maxRate
        losses = regRate * tf.log(self.y_) + (1 - regRate) * tf.log(1 - self.y_)
        loss = -tf.reduce_sum(losses)
        # regLoss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        # self.loss = loss + self.reg * regLoss
        self.loss = loss

    def add_train_step(self):
        '''
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.lr = tf.train.exponential_decay(self.lr, global_step,
                                             self.decay_steps, self.decay_rate, staircase=True)
        '''
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_step = optimizer.minimize(self.loss)

    def init_sess(self):
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.allow_soft_placement = True
        self.sess = tf.Session(config=self.config)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        if os.path.exists(self.checkPoint):
            [os.remove(f) for f in os.listdir(self.checkPoint)]
        else:
            os.mkdir(self.checkPoint)

    def run(self):
        best_hr = -1
        best_NDCG = -1
        best_epoch = -1
        print("Start Training!")
        for epoch in range(self.maxEpochs):
            print("="*20+"Epoch ", epoch, "="*20)
            self.run_epoch(self.sess)
            print('='*50)
            print("Start Evaluation!")
            hr, NDCG = self.evaluate(self.sess, self.topK)
            print("Epoch ", epoch, "HR: {}, NDCG: {}".format(hr, NDCG))
            if hr > best_hr or NDCG > best_NDCG:
                best_hr = hr
                best_NDCG = NDCG
                best_epoch = epoch
                self.saver.save(self.sess, self.checkPoint)
            if epoch - best_epoch > self.earlyStop:
                print("Normal Early stop!")
                break
            print("="*20+"Epoch ", epoch, "End"+"="*20)
        print("Best hr: {}, NDCG: {}, At Epoch {}".format(best_hr, best_NDCG, best_epoch))
        print("Training complete!")
        # self.metrics(self.sess)

    def run_epoch(self, sess, verbose=10):
        train_u, train_i, train_r = self.dataSet.getInstances(self.train, self.negNum)
        train_len = len(train_u)
        shuffled_idx = np.random.permutation(np.arange(train_len))
        train_u = train_u[shuffled_idx]
        train_i = train_i[shuffled_idx]
        train_r = train_r[shuffled_idx]

        num_batches = len(train_u) // self.batchSize + 1

        losses = []
        for i in range(num_batches):
            min_idx = i * self.batchSize
            max_idx = np.min([train_len, (i+1)*self.batchSize])
            train_u_batch = train_u[min_idx: max_idx]
            train_i_batch = train_i[min_idx: max_idx]
            train_r_batch = train_r[min_idx: max_idx]

            feed_dict = self.create_feed_dict(train_u_batch, train_i_batch, train_r_batch)
            _, tmp_loss = sess.run([self.train_step, self.loss], feed_dict=feed_dict)
            losses.append(tmp_loss)
            if verbose and i % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                    i, num_batches, np.mean(losses[-verbose:])
                ))
                sys.stdout.flush()
        loss = np.mean(losses)
        print("\nMean loss in this epoch is: {}".format(loss))
        return loss

    def create_feed_dict(self, u, i, r=None, drop=None):
        return {self.user: u,
                self.item: i,
                self.rate: r,
                self.drop: drop}

    def evaluate(self, sess, topK):
        def getHitRatio(ranklist, targetItem):
            for item in ranklist:
                if item == targetItem:
                    return 1
            return 0
        def getNDCG(ranklist, targetItem):
            # print('ranklist:',ranklist)
            # print('targetItem:', targetItem)
            for i in range(len(ranklist)):
                item = ranklist[i]
                if item == targetItem:
                    return math.log(2) / math.log(i+2)
            return 0

        hr =[]
        NDCG = []
        testUser = self.testNeg[0]
        testItem = self.testNeg[1]
        usr_scores = {}
        for i in range(len(testUser)): # 1078
            usr_idx = testUser[i][0]
            target = testItem[i][0]
            feed_dict = self.create_feed_dict(testUser[i], testItem[i])
            predict = sess.run(self.y_, feed_dict=feed_dict) # len 32
            # print('predict:', predict)
            usr_scores[str(usr_idx)] = list(predict)
            item_score_dict = {}
            for j in range(len(testItem[i])): #32
                item = testItem[i][j]
                item_score_dict[item] = predict[j]

            # ranklist = heapq.nlargest(topK, item_score_dict, key=item_score_dict.get)
            ranklist = heapq.nlargest(32, item_score_dict, key=item_score_dict.get)
            tmp_hr = getHitRatio(ranklist, target)
            # print(target)
            tmp_NDCG = getNDCG(ranklist, target)
            hr.append(tmp_hr)
            NDCG.append(tmp_NDCG)
        print('Len usr_scores:', len(usr_scores))
        write_json(str(usr_scores),'./Data/ours/usr_scores.json')
        return np.mean(hr), np.mean(NDCG)

    def metrics(self, sess):
        RS = []
        targets = []

        testUser = self.testNeg[0]
        testItem = self.testNeg[1]
        undup = []
        for i in range(len(testUser)): # 1078
            if testUser[i][0] in undup:
                continue
            
            undup.append(testUser[i][0])

            target = testItem[i]
            targets.append(target)

            feed_dict = self.create_feed_dict(testUser[i], testItem[i])
            predict = sess.run(self.y_, feed_dict=feed_dict) # len 32
            RS.append(predict)

        print('undup:', len(undup), undup)
        # print(len(ranklists), len(targets))
        Rss = np.asarray(RS)
        targets = np.asarray(targets)
        print(Rss.shape, targets.shape)

        usr_test_amount = 150
        sumtarget = len(testUser) # 1078
        testRS = Rss
        target = np.load('./Data/ours/target.npy')

        def F1_score(prec,rec):
            f1 = 2*((prec*rec)/(prec+rec))
            return f1

        def topN(RSls, n):
            maxn = np.argsort(RSls)[::-1][:n]
            return maxn
        all_sort = []

        for i in range(usr_test_amount):
            all_sort.append(topN(list(testRS[i]),len(testRS[i])))
            
        all_sort = np.asarray(all_sort)
        print(all_sort.shape)
        def DCG(prec_list): #找出前n名的[1,1,1,0,...]
            dcg = 0
            for i in range(len(prec_list)):
                dcg += (2**prec_list[i]-1)/math.log2(i+2)
            return dcg

        def NDCG(target, testRS, num_ndcg, all_sort): #target是真正的喜好
            total_ndcg = 0
            
            for m in range(usr_test_amount): # the number of testing users
                idcg = DCG(target[m][:num_ndcg])
                
                pre_list = []
                for s in all_sort[m][:num_ndcg]:
                    #print(m,s,target[m][s])
                    pre_list.append(target[m][s]) #把prec_list 的 score加進去
                
                dcg = DCG(pre_list)
                ndcg = dcg/idcg
                total_ndcg += ndcg
                
            avg_ndcg = total_ndcg/usr_test_amount
            return avg_ndcg

        from sklearn.metrics import average_precision_score

        def MAP(target,testRS):
            total_prec = 0
            for u in range(usr_test_amount):
                y_true = target[u]
                y_scores = testRS[u]
                total_prec += average_precision_score(y_true, y_scores)
                
            Map_value = total_prec/usr_test_amount
            
            return Map_value

        print('\n==============================\n')
        # Top N
        N = [1, 5]
        correct = 0

        for n in N:
            print('Top', n)
            correct = 0

            for i in range(len(testRS)):
                topn = topN(testRS[i], n)
                sum_target = int(np.sum(target[i]))
                TP = 0
                for i in topn:
                    if i < sum_target:
                        TP += 1

                correct += TP

            print('Num of TP:', correct)
            prec = correct/(len(testRS)*n) #150*n
            recall = correct/sumtarget

            print('prec:', prec)
            print('recall:', recall)
            try:
                print('F1_score:', F1_score(prec, recall))
            except ZeroDivisionError:
                print('ZeroDivisionError')
                pass
            print('*****')

        print('\n==============================\n')

        # NDCG
        num_ndcgs = [10]
        for num_ndcg in num_ndcgs:
            print('NDCG@', num_ndcg)
            print('NDCG score:', NDCG(target, testRS, num_ndcg, all_sort))
            print('*****')

        print('\n==============================\n')

        # MAP
        print('MAP:', MAP(target,testRS))
        print('\n==============================\n')
        
if __name__ == '__main__':
    main()