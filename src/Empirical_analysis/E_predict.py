#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# ------------------------------------------------------------------------
# Project: 2017 Seed cup round 2
# Author: Jiyang Qi, Zhihao Wang, Yue Pan
# GitHub: https://github.com/zxc479773533/2017-Seed-Cup-Round-2.git
# Module: Empirical analysis
# ------------------------------------------------------------------------

import os
import pickle
import numpy as np
import pandas as pd


def get_numlist(datafile, numlistfile, usernum):
    '''
    get number of behaviors of every user
    '''
    if os.path.exists(numlistfile):
        return np.load(numlistfile)
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
            np.save(numlistfile, numlist)
        return numlist


def get_user_level(userfile, levelfile):
    '''
    get users' level
    '''
    if os.path.exists(levelfile):
        return np.load(levelfile)
    else:
        userlevel = []
        with open(userfile, 'r') as rfp:

            # read line by line to save memory
            while True:
                rline = rfp.readline()

                # break if it's the end of the file
                if not rline:
                    break
                userlevel.append(int(rline.split(',')[1]))

            # save the numlist to pkl
            np.save(levelfile, userlevel)
        return userlevel


def calc_time_frame(user, boughtlist, behaviors, userlevel,
                    time_frame, time_bound=1503417600):
    '''
    get the time frame in which the user will buy something
    '''
    interval_cart = []  # time interval from adding it to cart to buying
    interval_star = []  # time interval from starring to buying

    # get the max time interval for every product
    for product in boughtlist:
        max_from_cart = 0
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
            interval_cart.append(max_from_cart)
        if max_from_star:
            interval_star.append(max_from_star)

    for product in behaviors:
        # if the user buy it more than one time
        if len(behaviors[product]) > 2:
            time_frame[user].append([1503676801, 1503676802, product])
            continue

        # if haven't buy it
        elif behaviors[product][0][2] == 0:

            # if the time adding or starring it is later than time_bound
            if behaviors[product][0][0] > time_bound or \
                    behaviors[product][0][1] > time_bound:
                # and if user have purchase history or
                # number of behaviors is less than 6
                if len(boughtlist) or len(behaviors) < 6:
                    time_frame[user].append([1503676801, 1503676802, product])
                    continue

            # if the time adding or starring it is later than time_bound or
            # user level is larger than 3
            elif userlevel[user] > 3 and behaviors[product][0][0] > time_bound or \
                    behaviors[product][0][1] > time_bound:
                # and if user have purchase history or
                # number of behaviors is less than 7
                if len(boughtlist) or len(behaviors) < 7:
                    time_frame[user].append([1503676801, 1503676802, product])
                    continue

            # calculate time frame
            product_range = [float("inf"), 0, product]
            if behaviors[product][0][0] and interval_star:
                product_range[0] = behaviors[product][0][0] + \
                    min(interval_star)
                product_range[1] = behaviors[product][0][0] + \
                    max(interval_star)
            if behaviors[product][0][1] and interval_cart:
                product_range[0] = min(product_range[0], behaviors[
                                       product][0][1] + min(interval_cart))
                product_range[1] = max(product_range[1], behaviors[
                                       product][0][1] + max(interval_cart))
            time_frame[user].append(product_range)


def get_data(datafile, numlistfile, usernum):
    '''
    read and get data to predict and return time frame to buy
    '''
    numlist = get_numlist(datafile, numlistfile, usernum)
    time_frame = [[] for i in range(usernum)]

    with open(datafile, 'r') as rfp:
        # ignore the head of table
        rfp.readline()
        for user in range(usernum):  # user = user_id - 1
            boughtlist = set()  # products bought
            behaviors = {}

            # get behavior info
            for i in range(numlist[user]):
                line = rfp.readline().split(',')
                behavior_type = int(line[4])
                product = int(line[2])
                if behavior_type > 1:
                    if product not in behaviors:
                        behaviors[product] = [[0, 0, 0]]
                    elif behaviors[product][-1][2]:
                        behaviors[product].append([0, 0, 0])
                    behaviors[product][-1][behavior_type - 2] = int(line[3])
                    if behavior_type == 4:
                        boughtlist.add(product)

            userlevel = get_user_level('../user_info.csv', 'user_class.npy')
            calc_time_frame(user, boughtlist,
                            behaviors, userlevel, time_frame)

    return time_frame


def filter(usernum, time_frame):
    '''
    filter time frame and get the (user,product) pair to predict
    '''
    result = []
    for user in range(usernum):
        for time in time_frame[user]:
            if time[0] >= 1503590400 and time[0] <= 1504022400 or \
                    time[1] >= 1503590400 and time[1] <= 1504022400:
                result.append([user + 1, time[2]])
    print('Data Length: {}'.format(len(result)))

    # save the result
    pickle.dump(result, open('result.pkl', 'wb'))
    result = pd.DataFrame(np.array(result))
    result.to_csv('result.csv')


if __name__ == '__main__':
    USERNUM = 62245
    datafile = '../user_behaviors.csv'
    numlistfile = 'numlist.npy'
    time_frame = get_data(datafile, numlistfile, USERNUM)
    filter(USERNUM, time_frame)
