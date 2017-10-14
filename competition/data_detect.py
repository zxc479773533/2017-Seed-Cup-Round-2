#!/usr/bin/env python
# coding=utf-8

import os
import numpy as np


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


def get_time(datafile, numlistfile, usernum):
    numlist = get_numlist(datafile, numlistfile, usernum)
    time_range = [[] for i in range(usernum)]

    with open('user_behaviors.csv', 'r') as rfp:
        rfp.readline()  # ignore the head
        right = 0
        allcount = 0
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

            for product in behaviors:
                bought_times = []
                other_times = []
                for behavior in behaviors[product]:
                    if (not behavior[0] == 0) and behavior[1] < 1503201600:
                        other_times.append(behavior[0])
                    if (not behavior[1] == 0) and behavior[1] < 1503201600:
                        other_times.append(behavior[1])
                    if not behavior[2] == 0:
                        bought_times.append(behavior[2])
                for other_time in other_times:
                    for bought_time in bought_times:
                        if bought_time > other_time + 3*86400 and bought_time < other_time + 6*86400:
                            right += 1
                allcount += len(other_times)
        print(right / allcount)
            # if user+1 > 3:
            #     break


if __name__ == '__main__':
    USERNUM = 62245
    datafile = 'user_behaviors.csv'
    numlistfile = 'numlist.npy'
    get_time(datafile, numlistfile, USERNUM)