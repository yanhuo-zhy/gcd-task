import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def clustering_loss(student_logits: torch.Tensor, 
                   teacher_logits: torch.Tensor, 
                   weight: float, 
                   teacher_temp=0.05, 
                   student_temp=0.1) -> torch.Tensor:
    """
    Compute clustering loss based on student and teacher outputs.
    
    Args:
        student_output (torch.Tensor): Student model's output.
        teacher_output (torch.Tensor): Teacher model's output.
        weight (float): Weight for the maximum entropy component.
        teacher_temp (float): Temperature for teacher softmax.
        student_temp (float): Temperature for student softmax/log-softmax.
        
    Returns:
        torch.Tensor: The clustering loss value.
    """

    # Normalize outputs by temperature
    student_out = student_logits / student_temp
    teacher_out = F.softmax(teacher_logits / teacher_temp, dim=-1).detach()

    # Calculate loss between teacher and student distributions
    total_loss = 0
    loss = -torch.sum(teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)
    total_loss = loss.mean()
    # Calculate entropy loss
    probs = F.softmax(student_logits / student_temp, dim=1)
    avg_probs = probs.mean(dim=0)
    entropy = -torch.sum(avg_probs * torch.log(avg_probs))
    # entrop_regularization = entropy - torch.log(torch.tensor(float(len(avg_probs))))
    # Combine the losses
    clustering_loss = total_loss - weight * entropy
    
    return clustering_loss


def unsupervised_contrastive_loss(features: torch.Tensor, 
                                  temperature=1.0, 
                                  device='cuda') -> torch.Tensor:
    """
    Compute the unsupervised contrastive loss.
    
    Args:
        features (torch.Tensor): Input features tensor.
        temperature (float): Temperature for scaling similarities.
        device (str): Device to move tensors to.
        
    Returns:
        torch.Tensor: The unsupervised contrastive loss value.
    """
    
    # Pre-compute some constants
    batch_size = features.size(0) // 2
    total_batch_size = 2 * batch_size
    
    
    # compute similarity matrix
    similarity_matrix = torch.matmul(features, features.T)
    
    # Compute masks for positive and negative pairs
    positive_mask = torch.zeros_like(similarity_matrix, dtype=torch.bool)
    positive_mask[torch.arange(batch_size), torch.arange(batch_size, total_batch_size)] = True
    
    negative_mask = torch.ones_like(similarity_matrix, dtype=torch.bool)
    negative_mask[batch_size:, :] = 0
    negative_mask[:, batch_size:] = 0
    negative_mask[torch.eye(total_batch_size, dtype=torch.bool)] = 0
    
    # Extract positive and negative pairs
    positives = similarity_matrix[positive_mask].view(-1, 1)
    negatives = similarity_matrix[negative_mask].view(batch_size, -1)
    
    # Concatenate and scale logits
    logits = torch.cat([positives, negatives], dim=1) / temperature
    
    # Compute the loss
    pseudo_labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
    loss = torch.nn.CrossEntropyLoss()(logits, pseudo_labels)
    
    return loss


def supervised_contrastive_loss(features: torch.Tensor, 
                                labels: torch.Tensor, 
                                temperature=0.07, 
                                base_temperature=0.07, 
                                device='cpu') -> torch.Tensor:
    """
    Compute the supervised contrastive loss.
    
    Args:
        features (torch.Tensor): features of shape [batch_size, feature_dim]
        labels (torch.Tensor): labels associated with features
        temperature (float): scaling temperature for contrastive loss
        base_temperature (float): base temperature for scaling
        
    Returns:
        torch.Tensor: computed loss
    """
    epsilon = 1e-7
    
    # Input validation
    if len(features.shape) != 2:
        raise ValueError('`features` needs to be [batch_size, feature_size]')
    if labels is None:
        raise ValueError('Please provide labels for supervised contrastive loss')
    
    batch_size = features.shape[0]
    labels = labels.contiguous().view(-1, 1)
    
    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')
    
    # Create the mask
    mask = torch.eq(labels, labels.T).float().to(device)
    
    # Compute anchor dot contrast
    similarity_matrix = torch.matmul(features, features.T) / temperature
    
    # Center the logits to prevent overflow
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()
    
    # Construct mask for valid logits
    eye_mask = torch.eye(batch_size, dtype=torch.bool).to(device)
    logits_mask = ~eye_mask
    mask = mask * logits_mask.float()
    
    # Compute log probabilities
    exp_logits = torch.exp(logits) * logits_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + epsilon)
    
    # Compute mean log probability for positives
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + epsilon)
    
    # Final loss computation
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    return loss.mean()




##------------------------------------------------GCD contrastive loss------------------------------------------------##
class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
def info_nce_logits(features, device):

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    # features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
 
    return logits, labels
