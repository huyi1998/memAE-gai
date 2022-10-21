import numpy as np
import os


def npy_to_log():
    # load_data = np.load("ckpt/Avenue_gt.npy", allow_pickle=True)
    load_data = np.load("ckpt/Avenue_gt.npy", allow_pickle=True)
    # for i in range(len(load_data)):
    #     print(len(load_data[i]))
    np.set_printoptions(threshold=np.inf)
    print(load_data.shape)
    print(load_data)
    # file_name = "C:/Users/klaus/Desktop/3.txt"
    # np.savetxt(file_name, load_data, delimiter=',', fmt='%s')
    print("done")


def log():
    file_name = "C:/Users/klaus/Desktop/log2.txt"
    arr = [[228, 208], [401, 100], [203, 185], [359, 320], [208, 187], [220, 130], [280, 224], [393, 358], [428, 363], [
        272, 175], [325, 290], [275, 239], [284, 240], [238, 197], [296, 249]]
    for tmp in arr:
        x = tmp[0]
        y = tmp[1]
        with open(file_name, "a") as filewrite:  # ”a"代表着每次运行都追加txt的内容
            # filewrite.write("[")
            for i in range(x):
                # if i % 40 == 0 and i != 0:
                #     filewrite.write('\n')
                if i < y:
                    filewrite.write(str(0) + " ")
                elif i < x - 1:
                    filewrite.write(str(1) + " ")
                else:
                    # filewrite.write(str(1) + "]")
                    filewrite.write(str(1))
            filewrite.write('\n')
            # filewrite.write('\n')


# 将txt文件读入numpy数组
def txt_to_numpy():
    file = open("C:/Users/klaus/Desktop/log2.txt")
    lines = file.readlines()
    # print(lines)
    # 初始化datamat
    data = []

    for line in lines:
        tmp = []
        # 写入datamat
        line = line.strip().split(' ')
        for i in line:
            tmp.append(i)
        a = np.array(tmp)
        a = a.astype(int)
        data.append(a)
    b = np.array(data, dtype=object)
    np.save("C:/Users/klaus/Desktop/log2.npy", b)


if __name__ == "__main__":
    load1=np.load("ckpt/factory3_gt原版.npy", allow_pickle=True)
    print(len(os.listdir("anomaly_data/factory/frames/testing")))
    # npy_to_log()
    # log()
    # txt_to_numpy()
