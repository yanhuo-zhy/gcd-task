import os
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, Subset


class CUBDataset(Dataset):
    def __init__(self, root_dir='./CUB_200_2011', transform=None, target_transform=None, train=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.imageloader = torchvision.datasets.folder

        if os.path.exists(root_dir):
            self._read_metadata()
        else:
            raise ValueError(f'Root directory {root_dir} does not exist.')


    # 数据集元数据读取函数
    def _read_metadata(self):
        # 读取images.txt文件，如果读取失败，抛出异常
        try:
            img_txt_file = open(os.path.join(self.root_dir, 'images.txt'))
            # 图片索引
            img_name_list = []
            for line in img_txt_file:
                # 最后一个字符为换行符
                img_name_list.append(line[:-1].split(' ')[-1])
        except:
            raise ValueError('File images.txt does not exist or is corrupted.')
        
        # 读取image_class_labels.txt文件，如果读取失败，抛出异常
        try:
            label_txt_file = open(os.path.join(self.root_dir, 'image_class_labels.txt'))
            # 标签索引，每个对应的标签减１，标签值从0开始
            label_list = []
            for line in label_txt_file:
                label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        except:
            raise ValueError('File image_class_labels.txt does not exist or is corrupted.')
        
        # 读取train_test_split.txt文件，如果读取失败，抛出异常
        try:
            train_test_file = open(os.path.join(self.root_dir, 'train_test_split.txt'))
            # 训练集索引
            train_test_list = []
            for line in train_test_file:
                train_test_list.append(int(line[:-1].split(' ')[-1]))
        except:
            raise ValueError('File train_test_split.txt does not exist or is corrupted.')
        
        # 是否为训练模式
        if self.train:
            self.data = [x for i, x in zip(train_test_list, img_name_list) if i]
            self.label = [x for i, x in zip(train_test_list, label_list) if i]
        else:
            self.data = [x for i, x in zip(train_test_list, img_name_list) if not i]
            self.label = [x for i, x in zip(train_test_list, label_list) if not i]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.label[idx]
        img_path = os.path.join(self.root_dir, 'images', self.data[idx])
        img = self.imageloader.default_loader(img_path)

        if self.transform:
            img = self.transform(img)
        
        if self.target_transform:
            label = self.target_transform(label)

        return img, label, idx


def split_dataset(dataset: Dataset, labeled_class: range, labeled_rate: float):
    """
    Split the dataset into labeled and unlabeled subsets based on the given labeled classes and rate.
    
    Args:
        dataset (Dataset): The input dataset to split.
        labeled_class (range): Range of classes that have labels.
        labeled_rate (float): Proportion of data in the labeled classes that is labeled.
        
    Returns:
        labeled_dataset (Dataset): Subset of the input dataset with labeled data.
        unlabeled_dataset (Dataset): Subset of the input dataset with unlabeled data.
    """
    np.random.seed(0)
    # Get indices of data that belong to the labeled classes
    labeled_class_indices = [i for i, (_, label, _) in enumerate(dataset) if label in labeled_class]
    
    # Randomly select a subset of the labeled class indices based on the labeled rate
    num_labeled = int(len(labeled_class_indices) * labeled_rate)
    labeled_indices = np.random.choice(labeled_class_indices, num_labeled, replace=False)
    
    # Get the unlabeled indices by finding the difference between all indices and the labeled indices
    all_indices = set(range(len(dataset)))
    unlabeled_indices = list(all_indices - set(labeled_indices))
    
    # Create the labeled and unlabeled datasets using the Subset class
    labeled_dataset = Subset(dataset, labeled_indices)
    unlabeled_dataset = Subset(dataset, unlabeled_indices)
    
    return labeled_dataset, unlabeled_dataset

# Example usage:
dataset = CUBDataset()
labeled_dataset, unlabeled_dataset = split_dataset(dataset, labeled_class=range(100), labeled_rate=0.5)
print(len(labeled_dataset))
print(len(unlabeled_dataset))