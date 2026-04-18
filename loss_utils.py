import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def entropy(logit):
    """计算整个 batch 的平均分布熵"""
    logit = logit.mean(dim=0)
    logit_ = torch.clamp(logit, min=1e-9)
    b = logit_ * torch.log(logit_)
    return -b.sum()



def consistency_loss(anchors, neighbors):
    """样本对级别的一致性损失"""
    b, n = anchors.size()
    similarity = torch.bmm(anchors.view(b, 1, n), neighbors.view(b, n, 1)).squeeze()
    ones = torch.ones_like(similarity)
    return F.binary_cross_entropy(similarity, ones)


def num_consistency_loss(anchors, neighbors):
    """cluster 级别的一致性"""
    anchors = anchors.t()
    neighbors = neighbors.t()
    similarity = torch.sum(anchors * neighbors, dim=1)
    return F.binary_cross_entropy_with_logits(similarity, torch.ones_like(similarity))

def compute_reliability(logits, temperature=1.0):
    """基于熵计算样本可靠性（低熵=高可靠性）"""
    eps = 1e-8
    probs = logits / (logits.sum(dim=1, keepdim=True) + eps)
    probs = torch.clamp(probs, min=eps)

    sample_entropy = -torch.sum(probs * torch.log(probs), dim=1)

    num_clusters = probs.size(1)
    max_entropy = np.log(num_clusters)
    normalized_entropy = sample_entropy / (max_entropy + eps)

    reliability = 1.0 - normalized_entropy
    reliability = reliability ** temperature

    return reliability


class StructureAlignmentLoss(nn.Module):
    """
    结构对齐损失 - 类似 DC_loss 结构

    """

    def __init__(self, temperature=0.5, eps=1e-8,use_reliability_weight=True):
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.use_reliability_weight = use_reliability_weight

    def forward(self, logits_text, logits_image, features_text, features_image):
        """
        Args:
            logits_text: [B, K] 文本聚类概率
            logits_image: [B, K] 图像聚类概率
            features_text: [B, D] 文本特征
            features_image: [B, D] 图像特征
        """
        batch_size = logits_text.size(0)

        # ========== 1. 计算可靠性 ==========
        if self.use_reliability_weight:
            reliability_text = compute_reliability(logits_text, self.temperature)
            reliability_image = compute_reliability(logits_image, self.temperature)
        else:
            reliability_text = compute_reliability(logits_text, 0)
            reliability_image = compute_reliability(logits_image, 0)

        # 联合可靠性（几何平均）
        joint_reliability = torch.sqrt(reliability_text * reliability_image)
        # ========== 2. 跨模态对比损失 ==========
        # 归一化特征
        features_text_norm = F.normalize(features_text, p=2, dim=1)
        features_image_norm = F.normalize(features_image, p=2, dim=1)

        # 相似度矩阵 [B, B]
        sim_matrix = torch.mm(features_text_norm, features_image_norm.t()) / 0.5

        # 正样本：对角线（同一样本的文本-图像）
        # 负样本：非对角线（不同样本的文本-图像）

        # 对比损失：正样本相似度高，负样本相似度低
        # 使用可靠性加权：高可靠性样本的正样本对齐更重要

        # 计算损失
        exp_sim = torch.exp(sim_matrix)

        # 正样本相似度
        pos_sim = torch.diag(exp_sim)  # [B]

        # 所有样本相似度（分母）
        all_sim = exp_sim.sum(dim=1)  # [B]

        # 对比损失
        contrastive_loss = -torch.log(pos_sim / (all_sim + self.eps) + self.eps)

        # 用可靠性加权（detach防止梯度流向可靠性）
        reliability_weight = joint_reliability.detach()
        loss_cross = (contrastive_loss * reliability_weight).sum() / (reliability_weight.sum() + self.eps)

        # ========== 3. 模态内聚集损失 ==========
        # 高可靠性样本引导同类聚集

        # 文本模态
        text_sim = torch.mm(features_text_norm, features_text_norm.t())/0.5
        pred_text = torch.argmax(logits_text, dim=1)
        mask_text = (pred_text.unsqueeze(1) == pred_text.unsqueeze(0)).float()
        mask_text = mask_text - torch.eye(batch_size, device=features_text.device)  # 排除自身

        if mask_text.sum() > 0:
            text_pos_sim = (torch.exp(text_sim) * mask_text).sum(dim=1)
            text_all_sim = torch.exp(text_sim).sum(dim=1)
            text_loss = -torch.log(text_pos_sim / (text_all_sim + self.eps) + self.eps)
            loss_text = (text_loss * reliability_text.detach()).sum() / (reliability_text.detach().sum() + self.eps)
        else:
            loss_text = torch.tensor(0.0, device=features_text.device)

        # 图像模态
        image_sim = torch.mm(features_image_norm, features_image_norm.t())/0.5
        pred_image = torch.argmax(logits_image, dim=1)
        mask_image = (pred_image.unsqueeze(1) == pred_image.unsqueeze(0)).float()
        mask_image = mask_image - torch.eye(batch_size, device=features_image.device)

        if mask_image.sum() > 0:
            image_pos_sim = (torch.exp(image_sim) * mask_image).sum(dim=1)
            image_all_sim = torch.exp(image_sim).sum(dim=1)
            image_loss = -torch.log(image_pos_sim / (image_all_sim + self.eps) + self.eps)
            loss_image = (image_loss * reliability_image.detach()).sum() / (reliability_image.detach().sum() +self.eps)
        else:
            loss_image = torch.tensor(0.0, device=features_image.device)

        # ========== 4. 总损失 ==========
        loss = 0.2*loss_cross + loss_text + loss_image

        # 统计信息
        with torch.no_grad():
            cross_sim = torch.diag(sim_matrix).mean()
            pred_agree = (pred_text == pred_image).float().mean()
            probs_text = logits_text / (logits_text.sum(dim=1, keepdim=True) + self.eps)
            probs_image = logits_image / (logits_image.sum(dim=1, keepdim=True) + self.eps)
            entropy_text = (-torch.sum(probs_text * torch.log(probs_text + self.eps), dim=1)).mean()
            entropy_image = (-torch.sum(probs_image * torch.log(probs_image + self.eps), dim=1)).mean()

        info_dict = {
            'reliability_text': reliability_text.mean().item(),
            'reliability_image': reliability_image.mean().item(),
            'entropy_text': entropy_text.item(),
            'entropy_image': entropy_image.item(),
            'cross_sim': cross_sim.item(),
            'pred_agreement': pred_agree.item(),
            'loss_cross': loss_cross.item(),
            'loss_text': loss_text.item(),
            'loss_image': loss_image.item(),
        }

        return loss, info_dict


class DataContrastiveLoss(nn.Module):
    """对比损失"""

    def __init__(self, base_temp=1.1,alpha=1.0, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.base_temp = base_temp

    def compute_base(self, x, y, curr_temp, batch_size):
        """计算基于相似度的样本权重"""
        features = torch.cat([x, y], dim=0)
        sim_matrix = torch.mm(features, features.T) / curr_temp

        pos_sim_ij = torch.diag(sim_matrix[:batch_size, batch_size:])
        pos_sim_ji = torch.diag(sim_matrix[batch_size:, :batch_size])
        pos_sim = torch.cat([pos_sim_ij, pos_sim_ji], dim=0)
        return torch.sigmoid(self.alpha * pos_sim)

    def forward(self, c_i, c_j, c_x, c_y, c_a, c_b, ifweights, ratio):
        batch_size = c_i.size(0)
        curr_temp = self.base_temp

        c_i = F.normalize(c_i, p=1, dim=1)
        c_j = F.normalize(c_j, p=1, dim=1)
        c_x = F.normalize(c_x, p=1, dim=1)
        c_y = F.normalize(c_y, p=1, dim=1)
        c_a = F.normalize(c_a, p=1, dim=1)
        c_b = F.normalize(c_b, p=1, dim=1)

        w_base1 = self.compute_base(c_a, c_b, curr_temp, batch_size)
        w_base2 = self.compute_base(c_x, c_y, curr_temp, batch_size)
        w_base = ratio * w_base1 + (1.0 - ratio) * w_base2

        features = torch.cat([c_i, c_j], dim=0)
        sim_matrix = torch.mm(features, features.T) / curr_temp

        labels = torch.arange(batch_size, device=c_i.device)
        labels = torch.cat([labels + batch_size, labels])

        logits = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)
        loss = -logits[torch.arange(2 * batch_size, device=c_i.device), labels]

        if ifweights:
            weighted_loss = (w_base * loss).sum() / (w_base.sum() + self.eps)
        else:
            weighted_loss = loss.mean()

        return weighted_loss
