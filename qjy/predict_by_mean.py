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


# load price of product
def get_price(productfile, pricefile):
    if os.path.exists(pricefile):
        return np.load(pricefile)
    else:
        pricelist = []
        with open(pricefile, 'r') as rfp:
            rfp.readline()
            while True:
                rline = rfp.readline()
                if not rline:
                    break
                pricelist.append(int(rline.split(',')[2]))
            np.save(pricefile, pricelist)
        return pricelist


# remove the largest and smallest
def remove_max_min(gaplist, length):
    if length > 4:
        gaplist.sort()
        gaplist = gaplist[1:-1]
        return length - 2
    else:
        return length


# calculate the mean_time
def calc_meantime(user, boughtlist, behaviors, mean_time):
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

    len_jiagou = len(gap_jiagou)
    len_star = len(gap_star)

    # remove the largest and smallest if length > 4
    len_jiagou = remove_max_min(gap_jiagou, len_jiagou)
    len_star = remove_max_min(gap_star, len_star)

    # get mean data
    mean_time[user][0] = (sum(gap_star) / len_star) if len_star else 0
    mean_time[user][1] = (
        sum(gap_jiagou) / len_jiagou) if len_jiagou else 0


# calculate the time_range
def calc_time_range(user, boughtlist, behaviors, time_range):
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

    # get mean data
    time_range[user][0] = (min(gap_star), max(gap_star))
    time_range[user][1] = (min(gap_jiagou), max(gap_jiagou))


# calculate the time to buy
def calc_time(user, behaviors, mean_time, time_to_buy):
    for product in behaviors:
        if behaviors[product][0][2] == 0:
            if behaviors[product][-1][0]:
                time_to_buy[user].append(
                    (product, behaviors[product][-1][0] + mean_time[user][0]))
            if behaviors[product][-1][1]:
                time_to_buy[user].append(
                    (product, behaviors[product][-1][1] + mean_time[user][1]))

    # print(user+1, gap_jiagou, mean_time[user])
    # print(time_to_buy[user])
    # print()


def get_mean_time(datafile, numlistfile, usernum):
    numlist = get_numlist(datafile, numlistfile, usernum)
    mean_time = np.array([[0, 0]] * usernum)
    time_to_buy = [[] for i in range(usernum)]

    with open('user_behaviors.csv', 'r') as rfp:
        rfp.readline()  # ignore the head
        for user in range(usernum):  # user = user_id - 1
            boughtlist = set()  # products bought
            behaviors = {}  # behaviors

            # get behavior info
            for i in range(numlist[user]):
                line = rfp.readline().split(',')
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

            # for product in behavior_info:

            calc_meantime(user, boughtlist, behaviors, mean_time)
            calc_time(user, behaviors, mean_time, time_to_buy)
            # if user+1 > 50:
            #     break

    return mean_time, time_to_buy


def filter(usernum, time_to_buy):
    result = []
    for user in range(usernum):
        for time in time_to_buy[user]:
            if time[1] > 1503676800 and time[1] < 1503936000:
                result.append([user+1, time[0]])
    result = pd.DataFrame(np.array(result))
    result.to_csv('result.csv')


if __name__ == '__main__':
    USERNUM = 62245
    datafile = 'user_behaviors.csv'
    numlistfile = 'numlist.npy'
    mean_time, time_to_buy = get_mean_time(datafile, numlistfile, USERNUM)
    filter(USERNUM, time_to_buy)
    mean_time = pd.DataFrame(mean_time)
    mean_time.to_csv('mean_time.csv')


# from 1500998400 to 1503676800
