#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# ------------------------------------------------------------------------
# Project: 2017 Seed cup round 2
# Author: Jiyang Qi, Zhihao Wang, Yue Pan
# GitHub: https://github.com/zxc479773533/2017-Seed-Cup-Round-2.git
# Module: Sort the data
# ------------------------------------------------------------------------

import pandas as pd
import numpy as np


# get users'behaviors
col = []
col.append('user')
col.append('product')
col.append('time')
col.append('do')

# sort the behaviors
behaviors = pd.read_csv('../behavior_info.csv').values
t1 = behaviors[:, ::-1].T
t2 = np.lexsort(t1)
behaviors = behaviors[t2]

# build data frame
user_behaviors = np.zeros((len(behaviors), 4))
for i in range(len(behaviors)):
    user_behaviors[i][0] = behaviors[i][0]
    user_behaviors[i][1] = behaviors[i][1]
    user_behaviors[i][2] = behaviors[i][2]
    user_behaviors[i][3] = behaviors[i][3]

user_behaviors = pd.DataFrame(user_behaviors, columns=col)
user_behaviors.to_csv('../user_behaviors.csv', float_format='%d')
