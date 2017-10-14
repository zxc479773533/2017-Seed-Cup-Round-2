#!/usr/bin/env python
# coding=utf-8

import os
import numpy as np
import pandas as pd


# load number of behavior for every user
def get_numlist(datafile, numlistfile, usernum):
    if os.path.exists(numlistfile):
        return np.load(numlistfile)
    else:
        numlist = [0] * usernum
        with open(datafile, 'r') as rfp:
            rfp.readline()
            while True:
                rline = rfp.readline()
                if not rline:  # if it's the end
                    break
                numlist[int(rline.split(',')[1])-1] += 1
            np.save(numlistfile, numlist)
        return numlist


def get_userclass(userfile, classfile):
    if os.path.exists(classfile):
        return np.load(classfile)
    else:
        userclass = []
        with open(userfile, 'r') as rfp:
            while True:
                rline = rfp.readline()
                if not rline:  # if it's the end
                    break
                userclass.append(int(rline.split(',')[1]))
            np.save(classfile, userclass)
        return userclass
        # calculate the time_range


def calc_time_range(user, boughtlist, behaviors, userclass, time_range, predict_time, time_bound=1503590400, superbound=1503504000, rate_bound=0.2):
    gap_jiagou = []  # time gap from jiagou to buying
    gap_star = []  # time gap from starring to buying
    for product in boughtlist:
        max_from_jiagou = 0  # get the max time gap for every product
        max_from_star = 0
        for every_buy in behaviors[product]:
            if every_buy[2]:
                if every_buy[1]:
                    max_from_jiagou = max(max_from_jiagou,
                                          every_buy[2] - every_buy[1])
                if every_buy[0]:
                    max_from_star = max(max_from_star,
                                        every_buy[2] - every_buy[0])
        if max_from_jiagou:
            gap_jiagou.append(max_from_jiagou)
        if max_from_star:
            gap_star.append(max_from_star)

    for product in behaviors:
        # print(product, len(behaviors[product]))
        if len(behaviors[product]) > 2:
            time_range[user].append([predict_time+1, predict_time+2, product])
            continue
        if behaviors[product][0][2] == 0:
            if behaviors[product][0][0] > time_bound or \
                    behaviors[product][0][1] > time_bound:
                if len(boughtlist) / len(behaviors) > rate_bound:
                    time_range[user].append([predict_time+1, predict_time+2, product])
                    continue
            elif userclass[user] > 4 and behaviors[product][0][0] > superbound or \
                    behaviors[product][0][1] > superbound:
                if len(boughtlist) / len(behaviors) > rate_bound:
                    time_range[user].append([predict_time+1, predict_time+2, product])
                    continue
            product_range = [float("inf"), 0, product]
            if behaviors[product][0][0] and gap_star:
                product_range[0] = behaviors[product][0][0] + min(gap_star)
                product_range[1] = behaviors[product][0][0] + max(gap_star)
            if behaviors[product][0][1] and gap_jiagou:
                product_range[0] = min(product_range[0], behaviors[
                                       product][0][1] + min(gap_jiagou))
                product_range[1] = max(product_range[1], behaviors[
                                       product][0][1] + max(gap_jiagou))
            time_range[user].append(product_range)

    # print(user+1, gap_jiagou, time_range[user])
    # print()


def get_time(datafile, numlistfile, usernum, predict_time):
    numlist = get_numlist(datafile, numlistfile, usernum)
    time_range = [[] for i in range(usernum)]

    with open('user_behaviors.csv', 'r') as rfp:
        rfp.readline()  # ignore the head
        for user in range(usernum):  # user = user_id - 1
            boughtlist = set()  # products bought
            behaviors = {}  # behaviors

            # get behavior info
            for i in range(numlist[user]):
                line = rfp.readline().split(',')
                if int(line[3]) > predict_time:
                    continue
                behavior_type = int(line[4])
                if behavior_type > 1:
                    product = int(line[2])
                    if product not in behaviors:
                        behaviors[product] = [[0, 0, 0]]
                    elif behaviors[product][-1][2]:
                        behaviors[product].append([0, 0, 0])
                    behaviors[product][-1][behavior_type-2] = int(line[3])
                    if behavior_type == 4:
                        boughtlist.add(product)

            userclass = get_userclass('user_info.csv', 'user_class.npy')
            calc_time_range(user, boughtlist, behaviors, userclass, time_range, predict_time, predict_time-86400, predict_time-2*86400)
            # if user+1 > 3:
            #     break

    return time_range


def filter(usernum, time_range, predict_time):
    result = []
    for user in range(usernum):
        for time in time_range[user]:
            if time[0] >= predict_time and time[0] <= predict_time+3*86400 or \
                    time[1] >= predict_time and time[1] <= predict_time+3*86400:
                result.append([user+1, time[2]])
    print(len(result))
    result = pd.DataFrame(np.array(result))
    result.to_csv('test.csv')


if __name__ == '__main__':
    USERNUM = 62245
    predict_time = 1503417600
    datafile = 'user_behaviors.csv'
    numlistfile = 'numlist.npy'
    time_range = get_time(datafile, numlistfile, USERNUM, predict_time)
    filter(USERNUM, time_range, predict_time)


# from 1500998400 to 1503676800
#107752
