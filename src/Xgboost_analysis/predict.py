#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import time
import random
import pickle
import numpy as np
import xgboost as xgb
import pandas as pd
from functools import reduce
from sklearn.model_selection import train_test_split

user_file = '../competition/user_info.csv'
product_file = '../competition/product_info.csv'
behavior_file = '../competition/user_behaviors.csv'
numlist_file = 'numlist.pkl'
USERNUM = 62245
start_date = '2017-7-26 00:00:00'
end_date = '2017-8-26 00:00:00'


def convert_age(age):
    if age < 0:
        return -1
    else:
        return age // 12


def to_time_stamp(timestr):
    timearray = time.strptime(timestr, "%Y-%m-%d %H:%M:%S")
    return int(time.mktime(timearray))


# load number of behavior for every user
def get_numlist(datafile, numlistfile, usernum):
    if os.path.exists(numlistfile):
        return pickle.load(open(numlistfile, 'rb'))
    else:
        numlist = [0] * usernum
        with open(datafile, 'r') as rfp:
            rfp.readline()
            while True:
                rline = rfp.readline()
                if not rline:  # if it's the end
                    break
                numlist[int(rline.split(',')[1])-1] += 1
            pickle.dump(numlist, open(numlistfile, 'wb'))
        return numlist


def get_user(userfile, dump_path):
    if os.path.exists(dump_path):
        return pickle.load(open(dump_path, 'rb'))
    else:
        user_info = []
        with open(userfile, 'r') as rfp:
            while True:
                rline = rfp.readline()
                if not rline:  # if it's the end
                    break
                rline = rline.split(',')
                user_info.append((int(rline[1]), convert_age(int(rline[6])),
                                  int(rline[7]) if int(rline[7]) > 0 else -1))
            pickle.dump(user_info, open(dump_path, 'wb'))
        return user_info


def get_product(productfile, dump_path):
    if os.path.exists(dump_path):
        return pickle.load(open(dump_path, 'rb'))
    else:
        product_info = []
        with open(productfile, 'r') as rfp:
            while True:  # read line by line to decrease memory usage
                rline = rfp.readline()
                if not rline:  # if it's the end
                    break
                rline = rline.split(',')
                product_info.append((int(rline[3]), int(rline[-1])))
            pickle.dump(product_info, open(dump_path, 'wb'))
        return product_info


def dict_count(dictionary, key, time):
    if key in dictionary:
        if time - dictionary[key][2] > 1800:
            dictionary[key][1] += 1
        dictionary[key][2] = time
        dictionary[key][0] += 1
    else:
        dictionary[key] = [1, 1, time]


def get_features(pos_dataset, neg_dataset, buy_label, user, boughtlist,
                 behaviors, browse_count, cart_count, star_count,
                 user_info, product_info, endtime):
    gap_cart = []  # time gap from carting to buying
    gap_star = []  # time gap from starring to buying
    for product in boughtlist:
        max_from_cart = 0  # get the max time gap for every product
        max_from_star = 0
        for every_buy in behaviors[product]:
            if every_buy[2]:
                if every_buy[1]:
                    max_from_cart = max(max_from_cart,
                                          every_buy[2] - every_buy[1])
                if every_buy[0]:
                    max_from_star = max(max_from_star,
                                        every_buy[2] - every_buy[0])
        if max_from_cart:
            gap_cart.append(max_from_cart)
        if max_from_star:
            gap_star.append(max_from_star)

    for product in behaviors:
        features = [len(behaviors[product]), len(boughtlist) / len(behaviors)]
        features += browse_count[product][:-1] if product in browse_count else [0, 0]
        features += cart_count[product][:-1] if product in cart_count else [0, 0]
        features += star_count[product][:-1] if product in star_count else [0, 0]
        features.append(1 if behaviors[product][-1][2] else 0)
        features.append(len(behaviors))
        features.append((endtime - max(behaviors[product][-1])) // 86400)
        features += list(user_info[user])
        features += list(product_info[product-1])
        if product in buy_label:
            pos_dataset['features'].append(features)
            pos_dataset['user_product'].append((user + 1, product))
        else:
            neg_dataset['features'].append(features)
            neg_dataset['user_product'].append((user + 1, product))

    # print(user+1, gap_cart, time_range[user])
    # print()


def get_data(datafile, numlist_file, usernum, starttime, endtime):
    pos_dataset = {'user_product': [], 'features': []}
    neg_dataset = {'user_product': [], 'features': []}
    pos_path = './cache/pos_set_%s_%s.pkl' % (starttime[5:9], endtime[5:9])
    neg_path = './cache/neg_set_%s_%s.pkl' % (starttime[5:9], endtime[5:9])
    if os.path.exists(pos_path) and os.path.exists(neg_path):
        pos_dataset = pickle.load(open(pos_path, 'rb'))
        neg_dataset = pickle.load(open(neg_path, 'rb'))
    else:
        starttime = to_time_stamp(starttime)
        endtime = to_time_stamp(endtime)

        with open(datafile, 'r') as rfp:
            rfp.readline()  # ignore the head
            numlist = get_numlist(datafile, numlist_file, usernum)
            user_info = get_user(user_file, 'user.pkl')
            product_info = get_product(product_file, 'product.pkl')
            for user in range(usernum):  # user = user_id - 1
                boughtlist = set()  # products bought
                behaviors = {}  # behaviors
                browse_count = {}
                star_count = {}
                cart_count = {}
                buy_label = []

                # get behavior info
                for i in range(numlist[user]):
                    line = rfp.readline().split(',')
                    product = int(line[2])
                    time = int(line[3])
                    behavior_type = int(line[4])
                    if time < starttime:
                        continue
                    elif time > endtime:
                        if behavior_type == 4 and time < endtime + 3*86400:
                            buy_label.append(product)
                        continue
                    if behavior_type > 1:
                        if product not in behaviors:
                            behaviors[product] = [[0, 0, 0]]
                        elif behaviors[product][-1][2]:
                            behaviors[product].append([0, 0, 0])
                        behaviors[product][-1][behavior_type-2] = time
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

        pickle.dump(pos_dataset, open(pos_path, 'wb'))
        pickle.dump(neg_dataset, open(neg_path, 'wb'))

    return pos_dataset, neg_dataset


def get_model(starttime, endtime, negnum, model_num):
    pos_dataset, neg_dataset = get_data(behavior_file, 'numlist.npy',
                                        USERNUM, starttime, endtime)
    pos_dataset = pos_dataset['features']
    neg_dataset = neg_dataset['features']
    poslen, neglen = len(pos_dataset), len(neg_dataset)
    print(poslen, neglen)
    bsts = []
    for i in range(model_num):
        neg_random = random.sample(neg_dataset, negnum)
        dataset = pos_dataset + neg_random
        label = [1] * poslen + [0] * negnum
        alldata = list(zip(dataset, label))
        random.shuffle(alldata)
        dataset, label = zip(*alldata)
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


def get_submit_model(starttime, endtime, negnum, model_num):
    pos_dataset, neg_dataset = get_data(behavior_file, 'numlist.npy',
                                        USERNUM, starttime, endtime)
    pos_dataset = pos_dataset['features']
    neg_dataset = neg_dataset['features']
    poslen, neglen = len(pos_dataset), len(neg_dataset)
    print(poslen, neglen)
    bsts = []
    for i in range(model_num):
        neg_random = random.sample(neg_dataset, negnum)
        dataset = pos_dataset + neg_random
        label = [1] * poslen + [0] * negnum
        alldata = list(zip(dataset, label))
        random.shuffle(alldata)
        dataset, label = zip(*alldata)
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
    result = []
    predict_label = []
    pos_dataset, neg_dataset = get_data(behavior_file, 'numlist.npy',
                                        USERNUM, starttime, endtime)
    data = xgb.DMatrix(neg_dataset['features'])
    for bst in bsts:
        predict_label.append(bst.predict(data))
    for i in range(len(neg_dataset['features'])):
        vote = 0
        for label in predict_label:
            if label[i] > 0.08:
                vote += 1
        if vote > model_num // 2:
            result.append(neg_dataset['user_product'][i])
    print(len(result))
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
