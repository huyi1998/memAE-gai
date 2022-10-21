import os
import shutil, glob
import sys


def fun1():
    # 设定文件路径
    path = 'dataset/factory/training/frames'
    # 对目录下的文件进行遍历
    for category in os.listdir(path):
        # 判断是否是文件
        if os.path.isdir(os.path.join(path, category)) == True:
            # 设置新文件名
            new_name = category.replace(category, category.split('-')[-1])
            # 重命名
            os.rename(os.path.join(path, category), os.path.join(path, new_name))
    # 结束
    print("End")


def fun2():
    outer_path = 'dataset/factory/training/frames'
    folderlist = os.listdir(outer_path)  # 列举文件夹
    for folder in folderlist:
        inner_path = os.path.join(outer_path, folder)
        filelist = os.listdir(inner_path)  # 列举图片
        for item in filelist:

            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(inner_path), item)  # 原图的地址
                dst = os.path.join(os.path.abspath(inner_path), item.split('_0')[-1])
                os.rename(src, dst)
    print("done")


# 移动文件
def fun3():
    sourcefile = 'dataset/factory/frames/training'
    # new_file=str(glob.glob(os.path.join(sourcefile,'*.txt')))
    Txt = glob.glob(os.path.join(sourcefile, '*.jpg'))
    for file in Txt:
        file_path = file.replace('\\', '/')
        new_dir = file_path.split('/')[0] + '/' + file_path.split('/')[1] + '/' + file_path.split('/')[3] + '/' + \
                  file_path.split('/')[2] + '/' + file_path.split('/')[4].split('_')[2]
        try:
            os.makedirs(new_dir)
        except:
            pass

        if file.split('\\')[1].split('_')[2] in new_dir:
            shutil.move(file_path, new_dir)
            continue


def count():
    path1 = 'E:/PythonProjects/MNAD/anomaly_data/factory/training/frames'
    path2 = 'dataset/factory/training/frames'
    print('testing')
    for p in os.listdir(path1):
        print(len(os.listdir(path1 + '/' + p)))
    print('training')
    for q in os.listdir(path2):
        print(len(os.listdir(path2 + '/' + q)))


def remove():
    path = 'E:/dataSet/dataSource/dataset/factory/training/frames'
    folder_list = os.listdir(path)
    for folder in folder_list:
        inner_path = os.path.join(path, folder)
        filelist = os.listdir(inner_path)  # 列举图片
        for item in filelist:
            if "training_video_" in item:
                os.remove(os.path.join(os.path.abspath(inner_path), item))


if __name__ == "__main__":
    # fun3()
    # fun2()
    # fun3()
    # count()
    remove()