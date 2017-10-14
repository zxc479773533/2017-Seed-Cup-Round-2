# ------------------------------------------------------------------------
# Project: 2017 Seed cup round 2
# Author: Jiyang Qi, Zhihao Wang, Yue Pan
# GitHub: https://github.com/zxc479773533/2017-Seed-Cup-Round-2.git
# Module: Simple tool to get behavior information
# ------------------------------------------------------------------------

import sys
import numpy as np
import pandas as pd


def get_browse(filename, action):
    """
    """
    data = pd.read_csv(filename)
    values = data.values
    print("Data shape: {}".format(values.shape))
    ret = {}
    for i in range(len(values)):
        if values[i][3] == action:
            if values[i][0] not in ret:
                ret[values[i][0]] = {values[i][1]: [1, values[i][2]]}
            else:
                if values[i][1] not in ret[values[i][0]]:
                    ret[values[i][0]][values[i][1]] = [1, values[i][2]]
                elif values[i][2] - ret[values[i][0]][values[i][1]][1] > 1800:
                    ret[values[i][0]][values[i][1]][0] += 1
                ret[values[i][0]][values[i][1]][1] = values[i][2]
    return ret


def get_information(browse_data, number):
    count = 0
    for user in browse_data:
        for product in browse_data[user]:
            if browse_data[user][product][0] > number:
                count += 1
    print(count)


if __name__ == '__main__':
    try:
        action = int(sys.argv[1])
        print("user behavior code: {}".format(action))
        behavior_info = '../behavior_info.csv'
        finall = get_browse(behavior_info, action)
        print(len(finall))
        get_information(finall, 1)
    except:
        print("Usage: python get_behavior_info [user behavior code]")