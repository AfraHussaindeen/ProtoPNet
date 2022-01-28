import torch
import cv2
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import settings as config


class ImageDataset(Dataset):
    def __init__(self, csv, train, test, transform):
        self.csv = csv
        self.train = train
        self.test = test
        self.image_names = []
        self.all_image_names = self.csv[:]['img']
        self.all_labels = np.array(self.csv.drop(['img'], axis=1))
        self.train_ratio = int(0.8 * len(self.csv))
        self.test_ratio = len(self.csv) - self.train_ratio
        self.label_names = csv.keys()[1::]
        self.class_to_idx = {self.label_names[i]: i for i in range(len(self.label_names))}
        self.idx_to_class = {i: self.label_names[i] for i in range(len(self.label_names))}

        # set the training data images and labels
        if self.train:
            print(f"Number of training images: {self.train_ratio}")
            self.image_names = list(self.all_image_names[:self.train_ratio])
            self.labels = list(self.all_labels[:self.train_ratio])
            self.transform = transform

        # set the validation data images and labels
        # elif self.train == False and self.test == False:
        #     print(f"Number of validation images: {self.valid_ratio}")
        #     self.image_names = list(self.all_image_names[-self.valid_ratio:-10])
        #     self.labels = list(self.all_labels[-self.valid_ratio:])
        #     # define the validation transforms
        #     self.transform = transforms.Compose([
        #         transforms.ToPILImage(),
        #         transforms.Resize((400, 400)),
        #         transforms.ToTensor(),
        #     ])

        elif self.test == True and self.train == False:
            self.image_names = list(self.all_image_names[self.train_ratio::])
            self.labels = list(self.all_labels[self.train_ratio::])
            # define the test transforms
            self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __convert_label_to_multi(self, label):
        new_label = []
        types = {
            0: 2,
            2: 3,
            5: 3,
            8: 3,
            11: 2,
            13: 3
        }

        l = {
            'bwv_absent': 0,
            'bwv_present': 1,
            'dag_absent': 0,
            'dag_irregular': 1,
            'dag_regular': 2,
            'pig_absent': 0,
            'pig_irregular': 1,
            'pig_regular': 2,
            'pn_absent': 0,
            'pn_atypical': 1,
            'pn_typical': 2,
            'rs_absent': 0,
            'rs_present': 1,
            'str_absent': 0,
            'str_irregular': 1,
            'str_regular': 2
        }
        init_ind = 0
        for k, v in types.items():
            for i in range(init_ind, init_ind + v):
                if label[i] == 1:
                    cat = self.idx_to_class[i]
                    new_label.append(l[cat])
            init_ind += v
        return new_label

    def __getitem__(self, index):
        image = cv2.imread(config.data_img_path + '/' + self.image_names[index])
        # convert the image from BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # apply image transforms
        image = self.transform(image)
        targets = self.labels[index]
        new_targets = self.__convert_label_to_multi(targets)
        print(torch.tensor(targets, dtype=torch.float32))
        return {
            # 'image': torch.tensor(image, dtype=torch.float32),
            'image': image.clone().detach(),
            'label': torch.tensor(new_targets, dtype=torch.float32),
            'key': self.image_names[index]
        }

# img_size = 224
# dataset = ImageDataset(pd.read_csv('one_hot_dataset.csv'), True, False, transform=transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize(size=(img_size, img_size)),
#     transforms.ToTensor(),
# ]))
# print(dataset.idx_to_class)
# print(dataset.__getitem__(4)['label'])