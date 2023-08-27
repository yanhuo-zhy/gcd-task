import os
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision import transforms

class CUBDataset(Dataset):
    def __init__(self, root_dir='./data/CUB_200_2011', transform=None, target_transform=None, train=True):
        """
        Args:
            root_dir (string): Directory with all images and metadata.
            transform (callable, optional): Transformation applied to the images.
            target_transform (callable, optional): Transformation applied to the labels.
            train (bool): Whether to load training data (True) or testing data (False).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.train_mode = train

        if os.path.exists(root_dir):
            self._load_metadata()
        else:
            raise ValueError(f'Root directory {root_dir} does not exist.')

    def _load_metadata(self):
        # Load image names from images.txt
        try:
            with open(os.path.join(self.root_dir, 'images.txt')) as img_file:
                img_names = [line.strip().split(' ')[-1] for line in img_file]
        except:
            raise ValueError('Error loading images.txt.')

        # Load image labels from image_class_labels.txt
        try:
            with open(os.path.join(self.root_dir, 'image_class_labels.txt')) as label_file:
                labels = [int(line.strip().split(' ')[-1]) - 1 for line in label_file]
        except:
            raise ValueError('Error loading image_class_labels.txt.')

        # Load train/test split from train_test_split.txt
        try:
            with open(os.path.join(self.root_dir, 'train_test_split.txt')) as split_file:
                split_flags = [int(line.strip().split(' ')[-1]) for line in split_file]
        except:
            raise ValueError('Error loading train_test_split.txt.')

        # Filter data based on training/testing mode
        if self.train_mode:
            self.image_paths = [img for flag, img in zip(split_flags, img_names) if flag]
            self.image_labels = [label for flag, label in zip(split_flags, labels) if flag]
        else:
            self.image_paths = [img for flag, img in zip(split_flags, img_names) if not flag]
            self.image_labels = [label for flag, label in zip(split_flags, labels) if not flag]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        label = self.image_labels[idx]
        img_path = os.path.join(self.root_dir, 'images', self.image_paths[idx])
        
        # Load the image using torchvision's default loader
        img = torchvision.datasets.folder.default_loader(img_path)

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

class DualAugmentation:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(2)]


class ImageTransforms:
    def __init__(self, image_size=32, interpolation=2, crop_pct=0.875):
        """Initialize with default or provided values."""
        self.image_size = image_size
        self.interpolation = interpolation
        self.crop_pct = crop_pct
        
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])

    def base_transforms(self):
        """
        Returns the base transformation list, including resizing, tensor conversion, 
        and normalization, which are common for both training and testing.
        """
        return [
            transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),
            # transforms.Resize((int(self.image_size / self.crop_pct), int(self.image_size / self.crop_pct)), self.interpolation)
        ]

    def train_transforms(self):
        """
        Returns the list of transformations for training, including data augmentations.
        This concatenates the base transforms at the end.
        """
        return self.base_transforms() + [ 
            transforms.RandomCrop(self.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)] 

    def test_transforms(self):
        """
        Returns the list of transformations for testing.
        This concatenates the base transforms at the end.
        """
        return self.base_transforms() + [
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)] 

    def get_transforms(self):
        """
        Returns the composed training and testing transformations.
        """
        train_transform = DualAugmentation(transforms.Compose(self.train_transforms()))
        test_transform = transforms.Compose(self.test_transforms())
        return train_transform, test_transform

def get_transform(args=None):
    """
    Given image_size and args (with interpolation and crop_pct attributes), this function 
    returns the training and testing transformations.
    """
    transform_obj = ImageTransforms(
        image_size=args.image_size if args and hasattr(args, 'image_size') else 224,
        interpolation=args.interpolation if args and hasattr(args, 'interpolation') else 2, 
        crop_pct=args.crop_pct if args and hasattr(args, 'crop_pct') else 0.875
    )
    return transform_obj.get_transforms()

class MergedDataset(Dataset):
    """
    Merges two datasets (labelled and unlabelled) allowing iteration over them in parallel.
    """

    def __init__(self, labelled_dataset, unlabelled_dataset, transform=None):
        """
        Initialize the MergedDataset with labelled and unlabelled datasets.

        Args:
        - labelled_dataset (Dataset): Dataset containing labelled samples.
        - unlabelled_dataset (Dataset): Dataset containing unlabelled samples.
        - transform (callable, optional): Transform to be applied on samples.
        """
        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.transform = transform

    def __len__(self):
        return len(self.labelled_dataset) + len(self.unlabelled_dataset)

    def __getitem__(self, index):    
        # Determine if the index falls within the labelled dataset
        if index < len(self.labelled_dataset):
            img, label, _ = self.labelled_dataset[index]
            labeled_or_not = np.array([1])
        else:
            # Adjust the index for the unlabelled dataset
            adjusted_index = index - len(self.labelled_dataset)
            img, label, _ = self.unlabelled_dataset[adjusted_index]
            labeled_or_not = np.array([0])

        # Apply the transformation, if provided
        if self.transform:
            img = self.transform(img)

        return img, label, labeled_or_not

class TransformedDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: callable = None):
        """
        Initialize the TransformedDataset with a dataset and an transform.
        
        Args:
            dataset (Dataset): The original dataset.
            transform (callable): A function/transform that takes an image
                and returns a transformed version.
        """
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx: int):
        img, label, uq_idx = self.dataset[idx]

        if self.transform:
            img = self.transform(img)

        return img, label, uq_idx

    def __len__(self) -> int:
        return len(self.dataset)