import torch
import numpy as np
from torch.utils.data import Dataset, Subset

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

# # Example usage:
# dataset = CUBDataset()
# labeled_dataset, unlabeled_dataset = split_dataset(dataset, labeled_class=range(100), labeled_rate=0.5)
