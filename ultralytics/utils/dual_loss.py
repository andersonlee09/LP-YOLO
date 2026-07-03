# utils/dual_loss.py
import torch
import torch.nn as nn
from ultralytics.utils.loss import PoseLoss, DetectionLoss


class DualTaskLoss:
    """双任务损失：Pose损失 + Detect损失"""

    def __init__(self, model):
        self.model = model

        # Pose损失（玉米穗）
        self.pose_criterion = PoseLoss(model)

        # Detect损失（气生根）
        self.det_criterion = DetectionLoss(model)

        # 任务权重
        self.pose_weight = 1.0
        self.det_weight = 1.0

    def __call__(self, preds, targets):
        """计算双任务损失
        Args:
            preds: (pose_pred, det_pred) 两个任务的预测
            targets: (pose_targets, det_targets) 两个任务的标签
        """
        pose_pred, det_pred = preds
        pose_targets, det_targets = targets

        # 计算Pose损失
        pose_loss = self.pose_criterion(pose_pred, pose_targets)

        # 计算Detect损失
        det_loss = self.det_criterion(det_pred, det_targets)

        # 加权总损失
        total_loss = self.pose_weight * pose_loss + self.det_weight * det_loss

        # 返回损失详情
        loss_items = {
            'pose_loss': pose_loss.detach(),
            'det_loss': det_loss.detach(),
            'total_loss': total_loss.detach()
        }

        return total_loss, loss_items