import numpy as np
import pandas as pd

# in
# all data
# out
# {user1:{product1:browse_count}}


def get_browse(filename, action):
    data = pd.read_csv(filename)
    values = data.values
    print(values.shape)
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
    filename = 'behavior_info.csv'
    finall = get_browse(filename, 3)
    print(len(finall))
    get_information(finall, 1)
