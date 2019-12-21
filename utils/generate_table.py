import numpy as np
import os
import csv

def gengrate_mask(data_dir, out_dir):
    mask_file = open(out_dir, 'w', newline='')
    w_m = csv.writer(mask_file)
    with open(data_dir) as csvfile:
    # with open(data_dir) as csvfile:
        reader = csv.reader(csvfile) # iterable object 每次调用next方法，读取一行
        headers = next(reader)
        #print (headers)
        w_m.writerow(headers) # 把第一行写出
        for row in reader:
            # 从第二行开始，
            # print(row) # row 是某一行 , row的数据结构是列表
            # mask_item = [] # 先把这一行归0
            w_flag = 1 # 如果这一行 的ID 和时间都没有，那么就舍去
            for i in range(len(row)): # 横向遍历？
                if i == 0:
                    if row[i] == '':
                        w_flag = 0
                        break
                    # mask_item.extend(row[i])
                    #print(mask_item[i])
                elif i >= 2:
                    # mask_item.extend([1 if row[i] != '' else 0])
                    # mask_item[i] = 1 if row[i] != '' else 0
                    row[i] = str(1) if row[i] != '' else str(0)
            print(row)
            if w_flag:
                #print (mask_item[:20])
                w_m.writerow(row)
    mask_file.close()


def gengrate_time(mask_dir, out_dir):
    interval_file = open(out_dir, 'w', newline='')
    w_t = csv.writer(interval_file)
    csvfile = open(mask_dir, 'r')
    reader = csv.reader(csvfile)
    headers = next(reader)
    # 首先输出变量名
    w_t.writerow(headers)

    row = next(reader)

    # 处理第一行
    # row  = next(reader)
    id_last = row[0]
    # 初始化deltatime
    dt_last = [0.0 for _ in range(len(row) - 2)]
    time_last = [int(row[1]) for _ in range(len(row) - 2)]
    m_last = [int(i) for i in row[2:]]
    shuchu = row[:2] + dt_last
    w_t.writerow(shuchu)

    for row in reader:

        # 读取第二行
        # row = next(reader)
        id_cur = row[0]
        # 如果两个id相同，说明是同一个人，那么就执行
        time_cur = [int(row[1]) for _ in range(len(row) - 2)]
        if id_cur == id_last:
            dt_cur = (np.array(time_cur) - np.array(time_last)) + (1 - np.array(m_last)) * np.array(dt_last)
            dt_cur = dt_cur.tolist()
            shuchu = row[:2] + dt_cur

            id_last = id_cur
            dt_last = dt_cur
            time_last = time_cur
            m_last = [int(i) for i in row[2:]]

            print(shuchu)
            w_t.writerow(shuchu)
        # 否则重新初始化
        else:
            id_last = row[0]
            dt_last = [0.0 for _ in range(len(row) - 2)]
            time_last = time_cur
            m_last = [int(i) for i in row[2:]]
            shuchu = row[:2] + dt_last
            print(shuchu)
            w_t.writerow(shuchu)

    interval_file.close()
    csvfile.close()


def count(dir):
    with open(dir) as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        print(headers)
        count = 0
        for row in reader:
            #print(row)
            count += 1
        print(count)


if __name__ == '__main__':

    import argparse
    par = argparse.ArgumentParser()
    par.add_argument('-x', type=str, help='Xs.csv')
    par.add_argument('-m', type=str, help='mask.csv')
    par.add_argument('-d', type=str, help='deltat.csv')
    args = par.parse_args()
    #需要生成的data路径
    data_dir = str(args.x)
    #保存mask的路径
    mask_dir = str(args.m)
    #保存time的路径
    time_dir = str(args.d)
    #先生成mask，再生成time
    gengrate_mask(data_dir, mask_dir)
    gengrate_time(mask_dir, time_dir)
    #count('total_mask.csv')
