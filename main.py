import os
import math
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy

from dataset import *
from util import *
from models import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(model, train_loader, test_loader, args, logger, writer):
    params_to_update = [param for param in model.parameters() if param.requires_grad]
    optimizer = SGD(
        params_to_update, 
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay
    )
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3
    )

    for epoch in range(args.epochs):
        total_loss, total_samples = 0.0, 0
        model.train()

        for batch_idx, (images, labels, labeled_mask) in enumerate(train_loader):
            labeled_mask = labeled_mask.view(-1).cuda().bool()
            images = torch.cat(images, dim=0).cuda()
            labels = labels.cuda()

            feature_proj, feature_logits = model(images)
            teacher_logits = feature_logits.detach().chunk(2)[1]

            # Calculate the supervised classification loss for labeled samples
            sup_logits = (feature_logits / 0.1).chunk(2)[0][labeled_mask]
            sup_labels = labels[labeled_mask]
            cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

            # Compute the clustering loss based on the student's and teacher's logits
            student_logits = feature_logits.chunk(2)[0]
            cluster_loss = clustering_loss(
                student_logits, teacher_logits, args.entropy_regularization_weight
            )
            # Replace clustering_loss with sharpen_clustering_loss
            # teacher_logits = feature_logits.detach().chunk(2)[0]
            # cluster_loss = sharpen_clustering_loss(
            #     student_logits, teacher_logits, args.entropy_regularization_weight
            # )

            #-----------------------------------Mine Representation Learning Loss----------------------------------#
            # Calculate self-supervised contrastive loss for all the samples
            con_loss = unsupervised_contrastive_loss(features=feature_proj, device=args.device)

            # Calculate supervised contrastive loss for the labeled samples
            sup_proj = feature_proj.chunk(2)[0][labeled_mask]
            sup_con_loss = supervised_contrastive_loss(sup_proj, sup_labels, device=args.device)
            #-----------------------------------Mine Representation Learning Loss----------------------------------#

            #-----------------------------------GCD Representation Learning Loss-----------------------------------#
            # # Calculate self-supervised contrastive loss for all the samples
            # con_logits, con_labels = info_nce_logits(features=feature_proj, device=args.device)
            # con_loss = torch.nn.CrossEntropyLoss()(con_logits, con_labels)

            # # Calculate self-supervised contrastive loss for all the samples
            # student_proj = torch.cat(
            #     [f[labeled_mask].unsqueeze(1) for f in feature_proj.chunk(2)], dim=1
            # )
            # sup_con_labels = labels[labeled_mask]
            # sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)
            #-----------------------------------GCD Representation Learning Loss-----------------------------------#

            combined_loss = 0
            combined_loss = args.loss_weight * (sup_con_loss + cls_loss)
            combined_loss += (1 - args.loss_weight) * (con_loss + cluster_loss)

            total_loss += combined_loss.item() * labels.size(0)
            total_samples += labels.size(0)

            optimizer.zero_grad()
            combined_loss.backward()
            optimizer.step()

            if batch_idx % args.print_epoch == 0:
                log_info = (
                    f'Epoch: [{epoch}][{batch_idx}/{len(train_loader)}]\t'
                    f'Loss: {combined_loss.item():.5f}\t'
                    f'Cls: {cls_loss.item():.4f}, Clust: {cluster_loss.item():.4f}, '
                    f'SupCon: {sup_con_loss.item():.4f}, Con: {con_loss.item():.4f}'
                )
                logger.info(log_info)

            iter_idx = epoch * len(train_loader) + batch_idx
            writer.add_scalars('Loss', {
                'cls_loss': cls_loss.item(),
                'cluster_loss': cluster_loss.item(),
                'sup_con_loss': sup_con_loss.item(),
                'con_loss': con_loss.item(),
                'total_loss': combined_loss.item()
            }, iter_idx)

        avg_loss = total_loss / total_samples
        logger.info(f'Train Epoch: {epoch} Avg Loss: {avg_loss:.4f}')
        writer.add_scalar('Average_Loss/avg_loss', avg_loss, epoch)

        logger.info('Testing...')
        all_acc, old_acc, new_acc = test(model, test_loader, args=args)
        logger.info(
            f'Train Accuracies: Overall Accuracy {all_acc:.4f} | Old Classes Accuracy {old_acc:.4f} | New Classes Accuracy {new_acc:.4f}'
        )

        writer.add_scalar('Accuracy/All', all_acc, epoch)
        writer.add_scalar('Accuracy/Old', old_acc, epoch)
        writer.add_scalar('Accuracy/New', new_acc, epoch)

        scheduler.step()

        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }

        model.save(path=args.model_path, save_dict=save_dict)
        logger.info(f"Model saved to {args.model_path}.")
        writer.close()


def test(model, test_loader, args):
    model.eval()
    model_device = args.device  

    total_preds, total_targets = [], []
    old_class_mask_list = []

    for _, (images, labels, _) in enumerate(tqdm(test_loader)):
        images = images.to(model_device)

        with torch.no_grad():
            _, logits = model(images)
            preds = logits.argmax(1).cpu().numpy()
            total_preds.append(preds)

            labels = labels.cpu().numpy()
            total_targets.append(labels)

            old_class_mask_list.extend([x in args.train_classes for x in labels])

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)
    old_class_mask = np.array(old_class_mask_list, dtype=bool)

    all_acc, old_acc, new_acc = calculate_clustering_accuracy(total_targets, total_preds, old_class_mask)

    return all_acc, old_acc, new_acc


def init_environment(args):
    if not args.evaluate:
        args, logger, tensorboard_writer = init_experiment(args)
        return args, logger, tensorboard_writer
    else:
        args, logger = init_experiment(args)
        return args, logger   
    

def create_datasets(args):
    train_transform, test_transform = get_transform(args=args)
    dataset = CUBDataset()
    # Split the dataset into labeled and unlabeled subsets
    labeled_dataset, unlabeled_dataset = split_dataset(
        dataset, labeled_class=args.train_classes, labeled_rate=0.5)
    # Merge the labeled and unlabeled subsets into a single dataset
    train_dataset = MergedDataset(
        labelled_dataset=deepcopy(labeled_dataset),
        unlabelled_dataset=deepcopy(unlabeled_dataset),
        transform=train_transform
    )
    # Create a test dataset
    test_dataset = TransformedDataset(
        dataset=deepcopy(unlabeled_dataset),
        transform=test_transform
    )
    return train_dataset, test_dataset


def get_data_loaders(train_dataset, test_dataset, args):
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    # Create a weighted sampler to sample from the train_dataset
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    train_loader = DataLoader(
        train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
        sampler=sampler, drop_last=True, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, num_workers=args.num_workers,
        batch_size=256, shuffle=False, pin_memory=False
    )
    return train_loader, test_loader


def evaluate(gcdmodel, test_loader, args, logger):
    all_acc, old_acc, new_acc = test(gcdmodel, test_loader, args)
    logger.info(f"Overall Accuracy: {all_acc}")
    logger.info(f"Old Classes Accuracy: {old_acc}")
    logger.info(f"New Classes Accuracy: {new_acc}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Replication GCD', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--experiment_name', default='Experiment-SharpenLoss-2')
    parser.add_argument('--grad_from_block', default=11) # vit_base: 11
    parser.add_argument('--print_epoch', default=1, type=int)
    parser.add_argument('--dataset_name', default='cub')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--evaluate', default=False, type=bool)
    parser.add_argument('--evaluate_path', default='/wang_hp/zhy/gcd-task/experiment/08-24-00-29-before-train', type=str)
    parser.add_argument('--config_path', default=None, type=str)
    
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--entropy_regularization_weight', type=float, default=2)
    parser.add_argument('--loss_weight', type=float, default=0.35)

    args = parser.parse_args()

    if not args.evaluate:
        args, logger, tensorboard_writer = init_environment(args)

        gcdmodel = GCDModel(num_classes=args.mlp_out_dim)
        # freeze the first 11 blocks of the backbone
        gcdmodel.freeze(num_layers_to_freeze=args.grad_from_block)
        gcdmodel = gcdmodel.to(args.device)
        logger.info('model build')

        train_dataset, test_dataset = create_datasets(args)
        train_loader, test_loader = get_data_loaders(train_dataset, test_dataset, args)

        train(gcdmodel, train_loader, test_loader, args=args, logger=logger, writer=tensorboard_writer)

    else:
        args, logger = init_environment(args)
        
        gcdmodel = GCDModel(num_classes=args.mlp_out_dim)
        gcdmodel.load(args.model_path)
        gcdmodel = gcdmodel.to(args.device)
        train_dataset, test_dataset = create_datasets(args)
        _, test_loader = get_data_loaders(train_dataset, test_dataset, args)
        evaluate(gcdmodel, test_loader, args, logger)