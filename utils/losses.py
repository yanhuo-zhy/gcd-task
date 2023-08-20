import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 自监督对比损失
# 这段代码定义了一个名为SupConLoss的类，该类用于计算监督对比损失。这种损失函数是监督对比学习中的关键部分，它鼓励来自同一类的样本在特征空间中靠近，而来自不同类的样本在特征空间中远离。
# 定义一个名为SupConLoss的类，用于计算监督对比损失。
class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""

    # 初始化函数，设置温度、对比模式和基础温度。
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()  # 调用父类torch.nn.Module的初始化函数
        self.temperature = temperature  # 设置温度
        self.contrast_mode = contrast_mode  # 设置对比模式
        self.base_temperature = base_temperature  # 设置基础温度

    # 定义前向传播函数，计算损失。
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

        # 确定使用的设备是CPU还是GPU。
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        # 检查特征的维度。
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        # 检查标签和掩码是否都被定义。
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        # 如果都没有定义，创建一个单位矩阵作为掩码。
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        # 如果定义了标签，根据标签创建掩码。
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        # 否则，使用给定的掩码。
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # 根据对比模式选择锚点特征。
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # 计算logits。
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # 为了数值稳定性，从logits中减去最大值。
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 创建掩码。
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # 计算log_prob。
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # 计算正样本的log-likelihood的均值。
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # 计算损失
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # 将损失重塑为锚点计数和批次大小的形状，并计算其均值。
        loss = loss.view(anchor_count, batch_size).mean()

        # 返回计算出的损失值。
        return loss

# 有监督对比损失

class SelfConLoss(torch.nn.Module):
    def __init__(self, temperature=1.0, n_views=2, device='cuda'):
        super(SelfConLoss, self).__init__()
        self.temperature = temperature
        self.n_views = n_views
        self.device = device

    def forward(self, features):
        # 计算批次大小的一半，因为每个样本有两个视图。
        b_ = 0.5 * int(features.size(0))
        
        # 创建标签，每个样本的两个视图都有相同的标签。
        labels = torch.cat([torch.arange(b_) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        # 对特征进行L2归一化。
        features = F.normalize(features, dim=1)
        
        # 计算特征之间的相似性矩阵。
        similarity_matrix = torch.matmul(features, features.T)

        # 从标签和相似性矩阵中去除主对角线。
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # 选择并组合多个正样本。
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        
        # 仅选择负样本。
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        # 将正样本和负样本的logits连接起来。
        logits = torch.cat([positives, negatives], dim=1)
        
        # 创建一个标签向量，其中所有正样本的标签为0。
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        # 通过温度参数调整logits。
        logits = logits / self.temperature
        
        # 使用交叉熵损失计算损失。
        loss = F.cross_entropy(logits, labels)
        
        return loss

# 聚类损失
# 这个DistillLoss类定义了知识蒸馏的损失函数，其中学生模型试图模仿教师模型的行为。损失是基于学生和教师模型的输出之间的交叉熵计算的。
# 定义一个类，表示蒸馏损失。这是用于知识蒸馏的损失函数，其中一个模型（学生）试图模仿另一个模型（教师）的行为。
# 这个DistillLoss类定义了知识蒸馏的损失函数，其中学生模型试图模仿教师模型的行为。损失是基于学生和教师模型的输出之间的交叉熵计算的。

class DistillLoss(nn.Module):
    def __init__(self, warmup_teacher_temp_epochs, nepochs, 
                 ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.1):
        super(DistillLoss, self).__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = self._create_temp_schedule(warmup_teacher_temp, teacher_temp, 
                                                                warmup_teacher_temp_epochs, nepochs)

    def _create_temp_schedule(self, warmup_temp, target_temp, warmup_epochs, total_epochs):
        """Create a temperature schedule for the teacher model."""
        warmup_schedule = np.linspace(warmup_temp, target_temp, warmup_epochs)
        stable_schedule = np.ones(total_epochs - warmup_epochs) * target_temp
        return np.concatenate((warmup_schedule, stable_schedule))

    def forward(self, student_output, teacher_output, epoch):
        student_out = student_output / self.student_temp
        student_out_chunks = student_out.chunk(self.ncrops)

        teacher_temp = self.teacher_temp_schedule[epoch]
        teacher_out_chunks = F.softmax(teacher_output / teacher_temp, dim=-1).detach().chunk(self.ncrops)

        total_loss = self._compute_loss(student_out_chunks, teacher_out_chunks)
        return total_loss

    def _compute_loss(self, student_chunks, teacher_chunks):
        """Compute the distillation loss between student and teacher outputs."""
        loss_values = []
        for iq, teacher_chunk in enumerate(teacher_chunks):
            for v, student_chunk in enumerate(student_chunks):
                if v == iq:
                    continue
                loss = torch.sum(-teacher_chunk * F.log_softmax(student_chunk, dim=-1), dim=-1)
                loss_values.append(loss.mean())
        return sum(loss_values) / len(loss_values)

