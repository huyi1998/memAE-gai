import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

rng = np.random.RandomState(2020)


class DataLoader(data.Dataset):
    def __init__(self, video_folder, transform, time_step=4, num_pred=1):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._time_step = time_step
        self._num_pred = num_pred

        if type(video_folder) is str:
            self.videos, video_string = setup(self.dir, self.videos)
        elif type(video_folder) is list:
            self.videos, video_string = setup_multiple(self.dir, self.videos)

        self.samples = self.get_all_samples(video_string)
        self.video_name = video_string

    def get_all_samples(self, video_string):
        frames = []
        for video in video_string:
            for i in range(len(self.videos[video]['frame']) - self._time_step*5):
                frames.append(self.videos[video]['frame'][i])
        return frames

    def load_image(self, filename):
        image = Image.open(filename)
        return image.convert('RGB')

    def __getitem__(self, index):
        frame_name = int(self.samples[index].split('/')[-1].split('_')[-1].split('.jpg')[0])
        if "venue" in self.dir or "venue" in self.dir[0]:
            video_name = self.samples[index].strip().split('frames/')[1].strip().split('_frame')[0]
        if "factory3" in self.dir or "factory3" in self.dir[0]:
            # video_name = self.samples[index].strip().split('frames/')[1].strip().split('_frame')[0]
            video_name = self.samples[index].strip().split('frames/')[1].strip().split('/t')[0]
        elif "UCSD" in self.dir:
            video_name = self.samples[index].split('/')[-1].split('_')[0]
            frame_name -= 1
        batch = []
        for i in range(self._time_step + self._num_pred):
            image = self.load_image(self.videos[video_name]['frame'][frame_name + i*5])
            if self.transform is not None:
                batch.append(self.transform(image))
        # return torch.cat(batch, 0) 同np.concatenate(batch, axis=0)
        return np.concatenate(batch, axis=0)

    def __len__(self):
        return len(self.samples)


def setup_multiple(path, videos):
    path_mom = path[0].strip().split('frames/')[0] + 'frames/'
    video_string_group = []
    video_filenames_group = []
    for single_path in path:
        _subpath = [single_path + v for v in os.listdir(single_path)]
        _video_string = np.unique([v.strip().split('frames/')[1].strip().split('_frame')[0] for v in _subpath])
        _video_string = sorted(_video_string, key=lambda s: int(s.strip().split('_')[-1]))
        video_string_group.append(_video_string)
        video_filenames_group.append(_subpath)
    video_string_group = [v for j in video_string_group for v in j]
    video_filenames_group = [v for j in video_filenames_group for v in j]
    for video in video_string_group:
        videos[video] = {}
        videos[video]['path'] = path_mom + video
        _subframe = [v for v in video_filenames_group if video + '_' in v]
        _subframe = sorted(_subframe, key=lambda s: int(s.strip().split('_')[-1].strip().split('.jpg')[0]))
        videos[video]['frame'] = np.array(_subframe)
        videos[video]['length'] = len(_subframe)
    return videos, video_string_group


def setup(path, videos):
    """Write all the images in a dict
    videos[video_string]['frame'] = (sorted image files) i.e., video_1_frame_1.jpg, video_1_frame_2.jpg,,,,video_1_frame_1000.jpg
    videos[video_string]['length'] = the total number of images in per video
    This function need to be manually edited based your own dataset structure
    """
    if "venue" in path or "factory3" in path:
        all_video_frames = []
        for folder in os.listdir(path):
            for image in os.listdir(path+folder):
                all_video_frames.append(path + folder + '/' + image)
        all_video_frames = np.array(all_video_frames)
        # all_video_frames = np.array([path + v for v in os.listdir(path)])
        # video_string = np.unique([v.strip().split('frames/')[1].strip().split('_frame')[0] for v in all_video_frames])
        video_string = np.unique([(path + v).split('frames/')[1].strip() for v in os.listdir(path)])
        video_string = sorted(video_string, key=lambda s: (s.strip().split('/')[-1]))
    elif "UCSDped" in path:
        video_string = np.unique([v.strip().split('_')[0] for v in os.listdir(path)])
        video_string = sorted(video_string)
        all_video_frames = np.array([path + v for v in os.listdir(path) if '.jpg' in v])
    print(video_string)
    if "venue" in path or "factory3" in path:
        path = path.strip().split('frames/')[0] + 'frames/'
    for video in video_string:
        videos[video] = {}
        videos[video]['path'] = path + video
        # _subframe = [v for v in all_video_frames if video + '_' in v]
        _subframe = [v for v in all_video_frames if video in v]
        if "venue" in path or "factory3" in path:
            _subframe = sorted(_subframe, key=lambda s: int(s.strip().split('_')[-1].strip().split('.jpg')[0]))
        else:
            _subframe = sorted(_subframe)
        videos[video]['frame'] = np.array(_subframe)

        videos[video]['length'] = len(_subframe)
    return videos, video_string


def give_frame_trans(dataset_type, imshape):
    height, width = imshape
    print("--There is no other augmentation except resizing, grayscale and normalization--")
    frame_trans = transforms.Compose([transforms.Resize([height, width]),  # 把给定的图片resize到given size
                                      # transforms.Grayscale(num_output_channels=1),  # 将图像转换为灰度图像
                                      transforms.ToTensor(),  # 将输入的数据shape H,W,C ——> C,H,W, 再将所有数除以255，将数据归一化到[0,1]
                                      transforms.Normalize([0.5], [0.5])])  # 用均值和标准差对张量图像进行归一化
    return frame_trans


def give_data_folder(dataset_type, dataset_path):
    """Give the path that saves the Avenue dataset or UCSDPed dataset
    Args:
        dataset_path: the main folder, in my case it's /project_scratch/bo/anomaly_data/
        dataset_type: Avenue/UCSDped2/UCSDped1/
    User defined!
    """
    if dataset_type == "Avenue" or dataset_type == "factory3":
        train_folder = dataset_path + dataset_type + '/frames/training/'
        test_folder = dataset_path + dataset_type + '/frames/testing/'

    elif dataset_type == "UCSDped2" or dataset_type == "UCSDped1":
        train_folder = dataset_path + dataset_type + "/Train_jpg/"
        test_folder = dataset_path + dataset_type + "/Test_jpg/"
    return train_folder, test_folder
