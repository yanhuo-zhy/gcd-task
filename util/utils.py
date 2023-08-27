import os
import logging
import argparse
import numpy as np
import torch
from datetime import datetime
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from scipy.optimize import linear_sum_assignment as linear_assignment
import configparser

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def create_experiment_directory(experiment_name):
    current_time = datetime.now().strftime('%m-%d-%H-%M')
    experiment_dir = os.path.join('experiment', f"{current_time}-{experiment_name}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    subdirs = ['logs', 'models', 'results', 'tensorboard_logs']
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
        
    return experiment_dir

def validate_evaluation_path(evaluate_path):
    essential_paths = [
        evaluate_path,
        os.path.join(evaluate_path, 'logs', 'log.txt'),
        os.path.join(evaluate_path, 'models', 'model.pth')
    ]
    for path in essential_paths:
        if not os.path.exists(path):
            raise ValueError(f"Invalid path: {path}. Please provide a valid path for evaluation using --evaluate_path")

def configure_logger(log_path):
    logging.basicConfig(filename=log_path,
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def modify_args_based_on_dataset(args):
    if args.dataset_name == 'cub':
        args.train_classes = range(100)
        args.num_labeled_classes = len(args.train_classes)
        args.unlabeled_classes = range(100, 200)
        args.num_unlabeled_classes = len(args.unlabeled_classes)
        args.interpolation = 3
        args.crop_pct = 0.875
        args.image_size = 224
        args.feat_dim = 768
        args.num_mlp_layers = 3
        args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

# def modify_args_based_on_config(args):
#     if args.config_path:
#         config = configparser.ConfigParser()
#         config.read(args.config)
#         for key, value in config['DEFAULT'].items():
#             if key in args:
#                 if isinstance(args.key, int):
#                     setattr(args, key, config.getint('DEFAULT', key))
#                 elif isinstance(args.key, float):
#                     setattr(args, key, config.getfloat('DEFAULT', key))
#                 else:
#                     setattr(args, key, value)

def configure_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    return device

def log_initial_info(logger, args):
    action = "Training" if not args.evaluate else "Evaluating"
    logger.info(f"{action} {args.experiment_name} with the following settings:")
    args_str = '\n '.join(f'{k}={v}' for k, v in vars(args).items())
    logger.info(f'Command-line arguments: {args_str}')

def init_tensorboard(tensorboard_log_dir, args):
    writer = SummaryWriter(tensorboard_log_dir)
    args_str = '\n '.join(f'{k}={v}' for k, v in vars(args).items())
    writer.add_text('Arguments', args_str)
    return writer


def init_experiment(args):
    # Setting Seeds
    set_seeds(args.seed)
    
    # Directory Management
    if not args.evaluate:
        experiment_dir = create_experiment_directory(args.experiment_name)
        args.log_path = os.path.join(experiment_dir, 'logs', 'log.txt')
        args.model_path = os.path.join(experiment_dir, 'models', 'model.pth')
        tensorboard_log_dir = os.path.join(experiment_dir, 'tensorboard_logs')
    else:
        validate_evaluation_path(args.evaluate_path)
        args.log_path = os.path.join(args.evaluate_path, 'logs', 'log.txt')
        args.model_path = os.path.join(args.evaluate_path, 'models', 'model.pth')
    
    # Logger Configuration
    logger = configure_logger(args.log_path)
    
    # Modify args based on dataset
    modify_args_based_on_dataset(args)
    # modify_args_based_on_config(args)  
    
    # Device Configuration
    args.device = configure_device()
    
    # Logging Initial Info
    log_initial_info(logger, args)
    
    # TensorBoard Initialization
    if not args.evaluate:
        writer = init_tensorboard(tensorboard_log_dir, args)
        return args, logger, writer

    return args, logger


def calculate_clustering_accuracy(y_true, y_pred, old_class_mask):
    """
    Calculate clustering accuracy: overall, old class, and new class accuracies.
    
    Arguments:
        y_true (np.array): Ground truth labels, shape `(n_samples,)`
        y_pred (np.array): Predicted labels, shape `(n_samples,)`
        old_class_mask (np.array): Boolean mask indicating if instance is from old class.

    Returns:
        tuple: Overall accuracy, old class accuracy, new class accuracy.
    """
    # Ensure the labels are integers
    y_true = y_true.astype(int)
    
    assert old_class_mask.dtype == np.bool_, "old_class_mask should be of boolean type!"
    assert len(y_true) == len(old_class_mask), "y_true and old_class_mask should have the same length!"

    # Get unique labels for old and new classes
    old_class_labels = set(y_true[old_class_mask])
    new_class_labels = set(y_true[~old_class_mask])

    # Ensure predictions and ground truth have same size
    assert y_pred.size == y_true.size
    
    # Compute confusion matrix
    num_labels = max(y_pred.max(), y_true.max()) + 1
    confusion_matrix = np.zeros((num_labels, num_labels), dtype=int)
    for i in range(y_pred.size):
        confusion_matrix[y_pred[i], y_true[i]] += 1

    # Find optimal class matchings
    label_matchings = linear_assignment(confusion_matrix.max() - confusion_matrix)
    label_matchings = np.vstack(label_matchings).T
    matching_dict = {j: i for i, j in label_matchings}
    
    # Compute overall accuracy
    total_correct = sum([confusion_matrix[i, j] for i, j in label_matchings])
    overall_accuracy = total_correct / y_pred.size

    # Compute old class accuracy
    old_correct = sum([confusion_matrix[matching_dict[i], i] for i in old_class_labels])
    old_accuracy = old_correct / sum(confusion_matrix[:, i].sum() for i in old_class_labels)

    # Compute new class accuracy
    new_correct = sum([confusion_matrix[matching_dict[i], i] for i in new_class_labels])
    new_accuracy = new_correct / sum(confusion_matrix[:, i].sum() for i in new_class_labels)

    return overall_accuracy, old_accuracy, new_accuracy


# if __name__ == "__main__":
    # args, logger, _, writer = init_experiment()
    # writer.close()