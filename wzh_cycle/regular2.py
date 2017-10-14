import pandas as pd
import numpy as np

# in
# numpy
# out
# {user1:[1,4,5],user2:[2,7,8]  }  数字代表数据在numpy里的位置
# 找出购买>1的用户
def ChooseUser(mat_data):
    ret = {}
    for i in range(len(mat_data)):
        if mat_data[i][0] in ret :
            ret[mat_data[i][0]].append(i)
        else:
            ret[mat_data[i][0]] = [i]
    temp = []
    for key in ret:
        if len(ret[key]) == 1:
            temp.append(key)
    for i in range(len(temp)):
        ret.pop(temp[i])
    return ret
# in 
# numpy and {user1:[1,4,5],user2:[2,7,8]}
# out 
# {user1:[1,4,5],user2:[2,7,8]} delete project which buny only once
# 删除 每种商品只购买一次的用户
def Clear_one(mat_data,user_data):
    for key in user_data:
        temp_count = [1]*len(user_data[key])
        for i in range(len(user_data[key])):
            for j in range(i + 1 , len(user_data[key])):
                if temp_count[i] == 0 or temp_count[j] == 0:
                    continue
                else:
                    if mat_data[user_data[key][i]][1] ==  mat_data[user_data[key][j]][1]:
                        temp_count[i] += 1
                        temp_count[j] = 0
        for i in range(len(user_data[key])):
            if temp_count[i] == 1:
                user_data[key][i] = 0
        while 0 in user_data[key]:
            user_data[key].remove(0)
    temp = []
    for key in user_data:
        if len(user_data[key]) == 0:
            temp.append(key)
    for i in range(len(temp)):
        user_data.pop(temp[i])
    count = 0
    for user in user_data:
        count += int(len(user_data[user]))
    print(count)        
    return user_data
# in
# {user1:[1,2,3]} 
# out
# {user1:{product1:[buy1,buy2,buy3],product1:[buy1,buy2,buy3]}   }
# 将 numpy 里的数据带入 buy1 等等代表时间
def User_product(mat_data,user_data):
    ret = {}
    for key in user_data:
        ret[key] = {}
        for i in range(len(user_data[key])):
            product = mat_data[user_data[key][i]][1]
            time = mat_data[user_data[key][i]][2]
            if product in ret[key]:
                ret[key][product].append(time)
            else:
                ret[key][product] = [time]
    return ret
# in 
# {user1:{product1:[buy1,buy2,buy3],product1:[buy1,buy2,buy3]}   }
# out 
# {user1:{product1:[average_time,average_number,min_time,min_number,max_time,max_number]}   }
# 得到user1在time时间内会购买number个product1 （平均）
def Get_time(product_time,once_buy_time):
    count = 0
    ret = {}
    for key in product_time:
        ret[key] = {}
        for product in product_time[key]:
            time_list = product_time[key][product]
            time_list.sort()
            time_different = []
            buy_number = []
            one_time = False
            for i in range(len(time_list)-1):
                if time_list[i+1] - time_list[i] > once_buy_time:
                    time_different.append(time_list[i+1] - time_list[i])
                    buy_number.append(1)
                    one_time = False 
                else:
                    if one_time == True:
                        buy_number[-1] += 1
                    else:
                        buy_number.append(1)
                        one_time = True
            count += len(buy_number)
            if len(time_different)!=0 and len(buy_number)!=0:
                ret[key][product] = [sum(time_different)/len(time_different),sum(buy_number)/len(buy_number),min(time_different),min(buy_number),max(time_different),max(buy_number)]
    print(count)
    return ret
# in  
# {user1:{product1:(time,number)}   }   {user1:{product1:[buy1,buy2,buy3],product1:[buy1,buy2,buy3]}   }
# out 
# {user1:{product1:count1,},}
# 进行预测 平均时间 产品时间（为了获得最后一次购买时间） 开始时间和结束时间 认定为一次购买的时间 目前为一天
def Predit(average_time,product_time,time_begin,time_end,once_buy_time):
    ret = {}
    for user in average_time:
        ret[user] = {}
        for product in average_time[user]:
            count = 1
            product_time[user][product].sort()
            time_list = product_time[user][product]
            for i in range(len(time_list)-1):
                if time_list[len(time_list)-i-1] - time_list[len(time_list)-i-2] < once_buy_time:
                    count += 1
                else:
                    break 
            if average_time[user][product][1] != 0:
                if (time_list[-1] + count/average_time[user][product][1]*average_time[user][product][0]) >= time_begin and \
                    (time_list[-1] + count/average_time[user][product][1]*average_time[user][product][0]) <= time_end:
                    ret[user][product] = average_time[user][product][1]
                elif (time_list[-1] + count/average_time[user][product][2]*average_time[user][product][3]) >= time_begin and \
                    (time_list[-1] + count/average_time[user][product][2]*average_time[user][product][3]) <= time_end:
                    ret[user][product] = average_time[user][product][3]
                elif (time_list[-1] + count/average_time[user][product][4]*average_time[user][product][5]) >= time_begin and \
                    (time_list[-1] + count/average_time[user][product][4]*average_time[user][product][5]) <= time_end:
                    ret[user][product] = average_time[user][product][5]
                else:
                    ret[user][product] = 0
            else:
                ret[user][product] = 0    

    return ret
# in 
# {user1:{product1:count1,},}
# out 
# numpy ((n,2))
# 将输出的字典转化为numpy
def out_put(ret_data):
    ret = np.zeros((1,2))
    for user in ret_data:
        for product in ret_data[user]:
            for i in range(int(ret_data[user][product]+0.5)):
                ret[-1][0] = user
                ret[-1][1] = product 
                ret = np.row_stack((ret,[0,0]))
    ret = ret[:-1]
    return ret

if __name__ == '__main__':
    mat_data = np.load('./data/buy_data.npy')
    user_data = ChooseUser(mat_data)
    user_data = Clear_one(mat_data,user_data)
    product_time = User_product(mat_data,user_data)
    average_time = Get_time(product_time,3600*24)
    ret_data = Predit(average_time,product_time,1503676800-3600*24,1503936000+3600*48,3600*24)
    finall = out_put(ret_data)
    print(finall.shape)
    col = range(2)
    row = range(len(finall))
    regular_data = pd.DataFrame(finall, index=row, columns=col)
    regular_data.to_csv('./data/regular_data2.csv')
