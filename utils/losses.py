import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_class_weights_from_counts(counts, power=0.5, min_weight=0.2):
    counts_t = torch.as_tensor(counts, dtype=torch.float32)
    counts_t = torch.clamp(counts_t, min=1.0)
    weights = counts_t.pow(-power)
    weights = weights / weights.mean()
    if min_weight is not None:
        weights = torch.clamp(weights, min=min_weight)
        weights = weights / weights.mean()
    return weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.register_buffer('weight', weight)

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        log_softmax = F.log_softmax(logits, dim=1)
        
        if self.label_smoothing > 0:
            smooth_val = self.label_smoothing / num_classes
            with torch.no_grad():
                smooth_target = torch.full_like(log_softmax, smooth_val)
                smooth_target.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing + smooth_val)
            ce_loss = -(smooth_target * log_softmax).sum(dim=1)
            probs = smooth_target.gather(1, targets.unsqueeze(1)).squeeze(1)
        else:
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
            probs = torch.exp(log_softmax).gather(1, targets.unsqueeze(1)).squeeze(1)
        
        focal_weight = (1 - probs) ** self.gamma
        loss = focal_weight * ce_loss
        
        if self.weight is not None:
            sample_weights = self.weight[targets]
            loss = loss * sample_weights
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class ArcFaceLoss(nn.Module):

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        s: float = 32.0,
        m: float = 0.30,
        label_smoothing: float = 0.0,
        class_weights=None,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.label_smoothing = label_smoothing
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = embeddings.float()

        x = F.normalize(embeddings, dim=1, eps=1e-6)
        w = F.normalize(self.weight.float(), dim=1, eps=1e-6)

        cosine = F.linear(x, w).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sine   = torch.sqrt(1.0 - cosine.pow(2))

        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        if self.label_smoothing > 0:
            smooth = self.label_smoothing / self.num_classes
            one_hot = one_hot * (1.0 - self.label_smoothing) + smooth

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        cw = self.class_weights.float() if self.class_weights is not None else None
        return F.cross_entropy(output, labels, weight=cw)

    @torch.no_grad()
    def get_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        x = F.normalize(embeddings.float(), dim=1, eps=1e-6)
        w = F.normalize(self.weight.float(), dim=1, eps=1e-6)
        return F.linear(x, w) * self.s