#!/usr/bin/env python
# coding=utf-8

import os
import pickle
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


def dict_count(dictionary, key, time):
    if key in dictionary:
        if time - dictionary[key][2] > 1800:
            dictionary[key][1] += 1
        dictionary[key][2] = time
        dictionary[key][0] += 1
    else:
        dictionary[key] = [1, 1, time]


def calc_score(browse, cart, star, product, lasttime):
    score = 0
    score += (browse[product][1] * 0.3) if product in browse else 0
    score += (cart[product][1] * 0.5) if product in cart else 0
    score += (star[product][1] * 0.4) if product in star else 0
    nonetime_score = 0
    nonetime_score += (browse[product][0] * 0.3) if product in browse else 0
    nonetime_score += (cart[product][0] * 0.5) if product in cart else 0
    nonetime_score += (star[product][0] * 0.4) if product in star else 0
    if lasttime > 1502985600:
        return score > 1.0 or nonetime_score > 1.3
    else:
        return score > 1.3 or nonetime_score > 1.6


def calc_time_range(user, boughtlist, behaviors, userclass, browse_count,
                    cart_count, star_count, time_range, time_bound=1503417600,
                    superbound=1503417600, rate_bound=0):
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
        if len(behaviors[product]) > 2:
            time_range[user].append([1503676801, 1503676802, product])
            continue
        elif calc_score(browse_count, cart_count, star_count, product, max(behaviors[product][-1][:-1])):
            if behaviors[product][-1][2] == 0:
                time_range[user].append([1503676801, 1503676802, product])
                continue
        elif behaviors[product][0][2] == 0:
            if behaviors[product][0][0] > time_bound or \
                    behaviors[product][0][1] > time_bound:
                if len(boughtlist) / len(behaviors) > rate_bound or len(behaviors) < 6:
                    time_range[user].append([1503676801, 1503676802, product])
                    continue
            elif userclass[user] > 3 and behaviors[product][0][0] > superbound or \
                    behaviors[product][0][1] > superbound:
                if len(boughtlist) / len(behaviors) > rate_bound or len(behaviors) < 6:
                    time_range[user].append([1503676801, 1503676802, product])
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


def get_time(datafile, numlistfile, usernum):
    numlist = get_numlist(datafile, numlistfile, usernum)
    time_range = [[] for i in range(usernum)]

    with open('user_behaviors.csv', 'r') as rfp:
        rfp.readline()  # ignore the head
        for user in range(usernum):  # user = user_id - 1
            boughtlist = set()  # products bought
            behaviors = {}  # behaviors
            browse_count = {}
            star_count = {}
            cart_count = {}

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
                    behaviors[product][-1][behavior_type-2] = int(line[3])
                    if behavior_type == 4:
                        boughtlist.add(product)
                    elif behavior_type == 3:
                        dict_count(cart_count, product, int(line[3]))
                    else:
                        dict_count(star_count, product, int(line[3]))
                else:
                    dict_count(browse_count, product, int(line[3]))

            userclass = get_userclass('user_info.csv', 'user_class.npy')
            calc_time_range(user, boughtlist, behaviors, userclass,
                            browse_count, cart_count, star_count, time_range)
            # if user+1 > 3:
            #     break

    return time_range


def filter(usernum, time_range):
    result = []
    for user in range(usernum):
        for time in time_range[user]:
            if time[0] >= 1503590400 and time[0] <= 1504022400 or \
                    time[1] >= 1503590400 and time[1] <= 1504022400:
                result.append([user+1, time[2]])
    print(len(result))
    pickle.dump(result, open('result.pkl', 'wb'))
    result = pd.DataFrame(np.array(result))
    result.to_csv('result.csv')


if __name__ == '__main__':
    USERNUM = 62245
    datafile = 'user_behaviors.csv'
    numlistfile = 'numlist.npy'
    time_range = get_time(datafile, numlistfile, USERNUM)
    filter(USERNUM, time_range)


# from 1500998400 to 1503676800
# 107752
