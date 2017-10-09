import pandas as pd
import numpy as np
# 得到所有的购买数据
def DeleteOther(filename):
    data = pd.read_csv(filename)
    values = data.values
    ret = np.zeros((1,4))
    for i in range(len(values)):
        if values[i][4] == 4:  # 是购买的话
            for j in range(4):  # 复制数据
                ret[-1][j] = values[i][j+1]
            ret = np.row_stack((ret,[0,0,0,0])) #增加一行
            print(ret.shape)
    ret = ret[:-1]
    return ret

if __name__ == '__main__':
    # 转化为numpy输出
    mat_data = DeleteOther('./data/user_behaviors.csv')
    col = range(4)
    row = range(len(mat_data)) 
    np.save('./data/buy_data.npy',mat_data) 
    buy_data = pd.DataFrame(mat_data, index=row, columns=col)
    buy_data.to_csv('./data/buy_data.csv')
