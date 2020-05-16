# -*- Encoding:UTF-8 -*-
import numpy as np
import sys
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
def write_json(data,dst_path):
    with open(dst_path, 'w') as outfile:
        json.dump(data, outfile, cls=NumpyEncoder)
def read_json(src_path):
    with open(src_path, 'r') as json_file:
        data = json.load(json_file)
    return data

class DataSet(object):
    def __init__(self, fileName):
        self.data, self.shape = self.getData(fileName)
        # self.train, self.test = self.getTrainTest()
        self.train = read_json('./Data/ours/'+fileName+'/train_t.json')
        self.test = read_json('./Data/ours/'+fileName+'/test_t.json')
        self.train_f = read_json('./Data/ours/'+fileName+'/train_f.json')
        self.test_f = read_json('./Data/ours/'+fileName+'/test_f.json')
        print('train/test:', len(self.train), len(self.test))
        print('train_f/test_f:', len(self.train_f), len(self.test_f))
        '''
        train/test: 20951 2495 
        train_f/test_f: 235221 2363
        '''
        self.trainDict = self.getTrainDict()

    def getData(self, fileName):
        print('getData...')

        data = []
        filePath = './Data/ours/'+fileName+'/myratings.dat'
        print(filePath)
        u = 0
        i = 0
        maxr = 0.0
        with open(filePath, 'r') as f:
            for line in f:
                if line:
                    lines = line[:-1].split(",")
                    user = int(lines[0])
                    movie = int(lines[1])
                    score = float(lines[2])
                    data.append((user, movie, score))
                    if user > u:
                        u = user
                    if movie > i:
                        i = movie
                    if score > maxr:
                        maxr = score
        self.maxRate = maxr
        print("Loading Success!\n"
                "Data Info:\n"
                "\tUser Num: {}\n"
                "\tItem Num: {}\n"
                "\tData Size: {}".format(u, i, len(data)))
        print([u, i])
        return data, [u, i]

    def getTrainTest(self):
        print('getTrainTest...')
        data = self.data
        # data = sorted(data, key=lambda x: (x[0], x[3]))
        train = []
        test = []
        for i in range(len(data)-1):
            user = data[i][0]-1
            item = data[i][1]-1
            rate = data[i][2]
            if data[i][0] != data[i+1][0]:
                test.append((user, item, rate))
            else:
                train.append((user, item, rate))

        test.append((data[-1][0]-1, data[-1][1]-1, data[-1][2]))

        print('train:', type(train), len(train), train[:10])
        print('test:', type(test), len(test), test[:10])
        '''
        train: <class 'list'> 21864 [(0, 2, 1.0), (0, 31, 1.0), (0, 36, 1.0), (0, 38, 1.0), (0, 55, 1.0), (0, 63, 1.0), (0, 96, 1.0), (0, 111, 1.0), (0, 120, 1.0), (1, 3, 1.0)]
        test: <class 'list'> 1582 [(0, 136, 1.0), (1, 163, 1.0), (2, 153, 1.0), (3, 161, 1.0), (4, 141, 1.0), (5, 161, 1.0), (6, 164, 1.0), (7, 164, 1.0), (8, 161, 1.0), (9, 156, 1.0)]
        '''
        return train, test

    #get train dict in (user,item) -> rating term
    def getTrainDict(self):
        print('getTrainDict...')
        dataDict = {}
        for i in self.train:
            dataDict[(i[0], i[1])] = i[2]
        return dataDict

    #the shape train matrix = [user_num, item_num], and only positive training data have rating
    def getEmbedding(self):
        print('getEmbedding...')
        train_matrix = np.zeros([self.shape[0], self.shape[1]], dtype=np.float32)
        for i in self.train:
            user = i[0]
            movie = i[1]
            rating = i[2]
            train_matrix[user][movie] = rating
        return np.array(train_matrix)

    #rating = 0 negative training data 
    def getInstances(self, data): #getInstances(self.train, self.negNum)
        print('getInstances...')
        user = []
        item = []
        rate = []
        for i in data:
            user.append(i[0])
            item.append(i[1])
            rate.append(i[2])

        for i in self.train_f:
            user.append(i[0])
            item.append(i[1])
            rate.append(i[2])

        print('Length:', len(user), len(item), len(rate)) #256172 256172 256172
        return np.array(user), np.array(item), np.array(rate)

    def getTestNeg(self, test_t):
        print('getTestNeg...')
        # Building test_truth
        test_userList = []
        test_truth = {}
        for sub_test in test_t:
            test_u = sub_test[0]
            test_i = sub_test[1]
            if test_u not in test_userList:
                test_userList.append(test_u)
                test_truth[str(test_u)] = [str(test_i)]
            else:
                test_truth[str(test_u)] = test_truth[str(test_u)]+[str(test_i)]
        write_json(test_truth, './Result/test_truth.json')

        user = []
        item = []
        finished_user = []
        for s in test_t:
            u = s[0]
            i = s[1]
            if u not in finished_user: #如果User 是新的
                #第二個以後變新的代表前面可以收成了
                if len(finished_user) > 0:
                    user.append(tmp_user)
                    tmp_item.reverse()
                    item.append(tmp_item)

                tmp_user = []
                tmp_item = []
                for fs in self.test_f:
                    fu = fs[0]
                    fi = fs[1]
                    if fu == u:
                        tmp_user.append(fu)
                        tmp_item.append(fi)
                tmp_user.append(u)
                tmp_item.append(i)
                finished_user.append(u)
                
            elif u in finished_user:
                tmp_user.append(u)
                tmp_item.append(i)

            if u == test_t[-1][0] and i == test_t[-1][1]: #如果是最後一個了話
                user.append(tmp_user)
                tmp_item.reverse()
                item.append(tmp_item)
        '''
        print('user:', len(user), user[:10])
        print('item:', len(item), item[:10])
        user: 150 [[30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30], [34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34], [41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41], [82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82], [127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127], [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128], [134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134, 134], [143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143], [160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160], [165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165]]
        item: 150 [[63, 68, 153, 7, 57, 164, 127, 65, 135, 82, 2, 47, 128, 102, 46, 101, 32, 28, 115, 132, 20, 0, 64, 59, 157, 25, 145, 19, 96, 143, 98, 14], [109, 55, 21, 13, 67, 88, 24, 69, 31, 115, 83, 141, 110, 47, 111, 107, 62, 78, 22, 19, 81, 137, 122, 58, 12, 91, 112, 156, 86, 14, 126, 28], [15, 63, 9, 90, 116, 75, 17, 57, 150, 115, 134, 39, 74, 38, 69, 162, 125, 104, 60, 143, 144, 64, 136, 81, 149, 120, 112, 85, 156, 95, 159, 147], [82, 3, 40, 86, 111, 39, 94, 139, 52, 148, 1, 85, 163, 60, 28, 152, 147, 22, 112, 6, 126, 10, 13, 98, 158, 99, 55, 96, 9, 53, 107, 58], [0, 92, 142, 5, 31, 105, 86, 69, 15, 21, 22, 141, 18, 158, 35, 137, 163, 47, 75, 155, 138, 37, 156, 91, 161, 66, 46, 101, 103, 28, 87, 112], [1, 15, 122, 108, 12, 41, 14, 153, 8, 147, 59, 155, 104, 48, 129, 47, 74, 89, 85, 111, 123, 6, 25, 117, 3, 36, 0, 62, 60, 121, 86, 84], [120, 55, 29, 127, 101, 125, 121, 140, 17, 30, 20, 118, 97, 32, 70, 56, 163, 131, 41, 8, 122, 72, 75, 142, 119, 46, 132, 19, 137, 146, 107, 147], [71, 52, 123, 146, 145, 65, 121, 18, 114, 0, 116, 78, 135, 87, 127, 48, 59, 89, 85, 25, 45, 90, 44, 14, 157, 117, 23, 4, 2, 144, 142, 83], [46, 17, 152, 25, 54, 21, 69, 118, 36, 73, 11, 122, 155, 91, 76, 60, 71, 77, 116, 39, 53, 13, 134, 133, 156, 88, 137, 138, 129, 68, 104, 112], [164, 158, 80, 114, 84, 21, 3, 163, 141, 24, 156, 136, 52, 125, 69, 31, 93, 17, 66, 96, 76, 40, 112, 107, 72, 94, 39, 83, 104, 85, 140, 0, 87, 105, 75, 116, 120, 12, 144, 138, 13, 99, 113]]
        '''
        return [np.array(user), np.array(item)]
