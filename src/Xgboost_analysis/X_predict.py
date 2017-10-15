#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# ------------------------------------------------------------------------
# Project: 2017 Seed cup round 2
# Author: Jiyang Qi, Zhihao Wang, Yue Pan
# GitHub: https://github.com/zxc479773533/2017-Seed-Cup-Round-2.git
# Module: Training data and predict
# ------------------------------------------------------------------------

import os
import time
import random
import pickle
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split

user_file = '../user_info.csv'
product_file = '../product_info.csv'
behavior_file = '../user_behaviors.csv'
numlist_file = 'numlist.pkl'
USERNUM = 62245


def convert_age(age):
    '''
    get babies' age
    '''
    if age < 0:
        return -1
    else:
        return age // 12


def to_time_stamp(time_str):
    '''
    convert time to time stamp
    '''
    time_array = time.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    return int(time.mktime(time_array))


def get_numlist(datafile, numlistfile, usernum):
    '''
    get number of behaviors of every user
    '''
    if os.path.exists(numlistfile):
        return pickle.load(open(numlistfile, 'rb'))
    else:
        numlist = [0] * usernum
        with open(datafile, 'r') as rfp:

            # read line by line to save memory
            rfp.readline()
            while True:
                rline = rfp.readline()

                # break if it's the end of the file
                if not rline:
                    break

                numlist[int(rline.split(',')[1]) - 1] += 1

            # save the numlist to pkl
            pickle.dump(numlist, open(numlistfile, 'wb'))
        return numlist


def get_user(userfile, dump_path):
    '''
    get users' level, babies' age and sex
    '''
    if os.path.exists(dump_path):
        return pickle.load(open(dump_path, 'rb'))
    else:
        user_info = []
        with open(userfile, 'r') as rfp:

            # read line by line to save memory
            while True:
                rline = rfp.readline()

                # break if it's the end of the file
                if not rline:
                    break

                rline = rline.split(',')
                user_info.append((int(rline[1]), convert_age(int(rline[6])),
                                  int(rline[7]) if int(rline[7]) > 0 else -1))

            # save data to pkl
            pickle.dump(user_info, open(dump_path, 'wb'))
        return user_info


def get_product(productfile, dump_path):
    '''
    get the type and price of every product
    '''
    if os.path.exists(dump_path):
        return pickle.load(open(dump_path, 'rb'))
    else:
        product_info = []
        with open(productfile, 'r') as rfp:

            # read line by line to save memory
            while True:
                rline = rfp.readline()

                # break if it's the end of the file
                if not rline:
                    break

                rline = rline.split(',')
                product_info.append((int(rline[3]), int(rline[-1])))

            # save data to pkl
            pickle.dump(product_info, open(dump_path, 'wb'))
        return product_info


def dict_count(dictionary, key, time):
    '''
    count the number of every behavior
    '''
    if key in dictionary:
        # if the time gap between two behaviors is less than half an hour
        if time - dictionary[key][2] > 1800:
            dictionary[key][1] += 1
        # save the time of this behavior
        dictionary[key][2] = time
        # count the number of every behavior
        dictionary[key][0] += 1
    else:
        dictionary[key] = [1, 1, time]


def get_features(pos_dataset, neg_dataset, buy_label, user, boughtlist,
                 behaviors, browse_count, cart_count, star_count,
                 user_info, product_info, endtime):
    '''
    get features and labels to train
    '''
    for product in behaviors:
        # get features
        features = [len(behaviors[product]), len(boughtlist) / len(behaviors)]
        features += browse_count[product][:-
                                          1] if product in browse_count else [0, 0]
        features += cart_count[product][:-
                                        1] if product in cart_count else [0, 0]
        features += star_count[product][:-
                                        1] if product in star_count else [0, 0]
        features.append(1 if behaviors[product][-1][2] else 0)
        features.append(len(behaviors))
        features.append((endtime - max(behaviors[product][-1])) // 86400)
        features += list(user_info[user])
        features += list(product_info[product - 1])

        # get labels
        if product in buy_label:
            pos_dataset['features'].append(features)
            pos_dataset['user_product'].append((user + 1, product))
        else:
            neg_dataset['features'].append(features)
            neg_dataset['user_product'].append((user + 1, product))


def get_data(datafile, numlist_file, usernum, starttime, endtime):
    '''
    get data to train
    '''
    pos_dataset = {'user_product': [], 'features': []}  # positive sample
    neg_dataset = {'user_product': [], 'features': []}  # negative sample

    # path to save cache
    pos_path = './cache/pos_set_%s_%s.pkl' % (starttime[5:9], endtime[5:9])
    neg_path = './cache/neg_set_%s_%s.pkl' % (starttime[5:9], endtime[5:9])
    if os.path.exists(pos_path) and os.path.exists(neg_path):
        pos_dataset = pickle.load(open(pos_path, 'rb'))
        neg_dataset = pickle.load(open(neg_path, 'rb'))
    else:
        starttime = to_time_stamp(starttime)
        endtime = to_time_stamp(endtime)

        with open(datafile, 'r') as rfp:

            # ignore the head of table
            rfp.readline()
            numlist = get_numlist(datafile, numlist_file, usernum)
            user_info = get_user(user_file, 'user.pkl')
            product_info = get_product(product_file, 'product.pkl')
            for user in range(usernum):  # user = user_id - 1
                boughtlist = set()  # products bought
                behaviors = {}  # behaviors
                browse_count = {}  # number of times browsing the product
                star_count = {}  # number of times starring the product
                cart_count = {}  # number of times adding the product to cart
                buy_label = []  # bought products to generate label

                # get behavior info
                for i in range(numlist[user]):
                    line = rfp.readline().split(',')
                    product = int(line[2])
                    time = int(line[3])
                    behavior_type = int(line[4])

                    if time < starttime:
                        continue
                    elif time > endtime:
                        if behavior_type == 4 and time < endtime + 3 * 86400:
                            buy_label.append(product)
                        continue

                    if behavior_type > 1:
                        # if behavior isn't browse
                        if product not in behaviors:
                            behaviors[product] = [[0, 0, 0]]
                        elif behaviors[product][-1][2]:
                            behaviors[product].append([0, 0, 0])

                        # get the behavior type and time
                        behaviors[product][-1][behavior_type - 2] = time
                        if behavior_type == 4:
                            boughtlist.add(product)
                        elif behavior_type == 3:
                            dict_count(cart_count, product, time)
                        else:
                            dict_count(star_count, product, time)
                    else:
                        dict_count(browse_count, product, time)

                get_features(pos_dataset, neg_dataset, buy_label,
                             user, boughtlist, behaviors,
                             browse_count, cart_count, star_count,
                             user_info, product_info, endtime)

        # save data to pkl
        pickle.dump(pos_dataset, open(pos_path, 'wb'))
        pickle.dump(neg_dataset, open(neg_path, 'wb'))

    return pos_dataset, neg_dataset


def get_model(starttime, endtime, neg_num, model_num):
    '''
    train and get the models to predict
    '''
    pos_dataset, neg_dataset = get_data(behavior_file, 'numlist.pkl',
                                        USERNUM, starttime, endtime)
    pos_dataset = pos_dataset['features']
    neg_dataset = neg_dataset['features']
    poslen, neglen = len(pos_dataset), len(neg_dataset)
    print(poslen, neglen)
    bsts = []  # list to save the trained bst models

    for i in range(model_num):
        neg_random = random.sample(neg_dataset, neg_num)
        dataset = pos_dataset + neg_random
        label = [1] * poslen + [0] * neg_num

        # shuffle the data to train
        alldata = list(zip(dataset, label))
        random.shuffle(alldata)
        dataset, label = zip(*alldata)

        # split dataset to train set and test set
        X_train, X_test, y_train, y_test = train_test_split(dataset,
                                                            label)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        param = {'max_depth': 3, 'eta': 0.09, 'silent': 1, 'nthread': 4,
                 'objective': 'binary:logistic'}
        num_round = 100
        param['eval_metric'] = "auc"
        plst = list(param.items())
        plst += [('eval_metric', 'logloss')]
        evallist = [(dtrain, 'train'), (dtest, 'eval')]
        bsts.append(xgb.train(plst, dtrain, num_round, evallist))

    return bsts


def get_submit_model(starttime, endtime, neg_num, model_num):
    '''
    get models to submit, without test data
    '''
    pos_dataset, neg_dataset = get_data(behavior_file, 'numlist.pkl',
                                        USERNUM, starttime, endtime)
    pos_dataset = pos_dataset['features']
    neg_dataset = neg_dataset['features']
    poslen, neglen = len(pos_dataset), len(neg_dataset)
    print(poslen, neglen)
    bsts = []  # list to save the trained bst models

    for i in range(model_num):
        neg_random = random.sample(neg_dataset, neg_num)
        dataset = pos_dataset + neg_random
        label = [1] * poslen + [0] * neg_num

        # shuffle the data to train
        alldata = list(zip(dataset, label))
        random.shuffle(alldata)
        dataset, label = zip(*alldata)

        # split dataset to train set and test set
        dtrain = xgb.DMatrix(np.array(dataset), label=np.array(label))
        param = {'max_depth': 3, 'eta': 0.09, 'silent': 1, 'nthread': 4,
                 'objective': 'binary:logistic'}
        num_round = 100
        param['eval_metric'] = "auc"
        plst = list(param.items())
        plst += [('eval_metric', 'logloss')]
        evallist = [(dtrain, 'train')]
        bsts.append(xgb.train(plst, dtrain, num_round, evallist))

    return bsts


def predict(bsts, starttime, endtime, model_num):
    '''
    predict the result with the models above
    '''
    result = []
    predict_label = []
    pos_dataset, neg_dataset = get_data(behavior_file, 'numlist.pkl',
                                        USERNUM, starttime, endtime)
    data = xgb.DMatrix(neg_dataset['features'])

    # use every bst model to predict
    for bst in bsts:
        predict_label.append(bst.predict(data))

    # vote to get the result
    for i in range(len(neg_dataset['features'])):
        vote = 0
        for label in predict_label:
            if label[i] > 0.083:
                vote += 1
        if vote > model_num // 2:
            result.append(neg_dataset['user_product'][i])

    # save the result
    print('Xgboost predict length: {}'.format(len(result)))
    pickle.dump(result, open('result.pkl', 'wb'))
    result = pd.DataFrame(np.array(result))
    result.to_csv('result.csv')


if __name__ == '__main__':
    model_num = 20
    # bsts = get_model("2017-7-26 00:00:00",
    #                  "2017-8-23 00:00:00", 5000, model_num)
    bsts = get_submit_model("2017-7-26 00:00:00",
                            "2017-8-23 00:00:00", 5000, model_num)
    predict(bsts, "2017-7-29 00:00:00", "2017-8-26 00:00:00", model_num)

    # merge E_predict result and X_predict result
    final_result = pickle.load(open('result.pkl', 'rb'))
    final_result += pickle.load(open('../Empirical_analysis/result.pkl', 'rb'))

    # remove the same pairs
    print('removing the same pairs')
    final_result = list(set([tuple(user_prod) for user_prod in final_result]))

    # save the final result
    print('final length: ', len(final_result))
    with open('../../answer.txt', 'w') as wfp:
        for user_product in final_result:
            wfp.write(str(user_product[0]) + '\t' +
                      str(user_product[1]) + '\n')
