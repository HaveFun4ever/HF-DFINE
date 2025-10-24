"""
DEIM: DETR with Improved Matching for Fast Convergence 
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)  
Copyright (c) 2024 D-FINE Authors. All Rights Reserved. 
"""

import torch
import torch.nn as nn
import torch.distributed    
import torch.nn.functional as F
import torchvision
import math
import copy
     
from .dfine_utils import bbox2distance
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou    
from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized   
from ..core import register
def shape_iou(box1, box2, xywh=True, scale=1, eps=1e-7,ratio=1.15):
    (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
    w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
 
    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    # Inner box IoU
    inner_b1_x1, inner_b1_x2, inner_b1_y1, inner_b1_y2 = x1 - w1_*ratio, x1 + w1_*ratio,\
                                                             y1 - h1_*ratio, y1 + h1_*ratio
    inner_b2_x1,inner_b2_x2, inner_b2_y1, inner_b2_y2 = x2 - w2_*ratio, x2 + w2_*ratio,\
                                                             y2 - h2_*ratio, y2 + h2_*ratio
    inner_inter = (torch.min(inner_b1_x2, inner_b2_x2) - torch.max(inner_b1_x1, inner_b2_x1)).clamp(0) * \
                   (torch.min(inner_b1_y2, inner_b2_y2) - torch.max(inner_b1_y1, inner_b2_y1)).clamp(0)
    inner_union = w1*ratio * h1*ratio + w2*ratio * h2*ratio - inner_inter + eps
    inner_iou = inner_inter/inner_union
    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps
    area_x1 = torch.min(b1_x1, b2_x1)
    area_y1 = torch.min(b1_y1, b2_y1)
    area_x2 = torch.max(b1_x2, b2_x2)
    area_y2 = torch.max(b1_y2, b2_y2)
    area= (area_x2 - area_x1) * (area_y2 - area_y1) + eps
    # IoU
    iou = inter / union
    g= ((area - union) / area)
    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance  
    ww = 2 * torch.pow(w2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
    hh = 2 * torch.pow(h2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
    c2 = cw ** 2 + ch ** 2 + eps                            # convex diagonal squared
    center_distance_x = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2) / 4
    center_distance_y = ((b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
    center_distance = hh * center_distance_x + ww * center_distance_y
    distance = center_distance / c2
    # aspect_ratio1 = w1/h1
    # aspect_ratio2 = w2/h2
    # ar_loss = 0.5*torch.abs(aspect_ratio1 - aspect_ratio2) / ((aspect_ratio1 + aspect_ratio2 + eps)/2)
    omiga_w = hh * torch.abs(w1 - w2) / torch.max(w1, w2)
    omiga_h = ww * torch.abs(h1 - h2) / torch.max(h1, h2)
    arctan = torch.atan(w2 / h2) - torch.atan(w1/ h1)
    v = (4.0 / math.pi ** 2) * (arctan ** 2)
    alpha = v / (1 - iou + v)
    closs=alpha * v
    shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
    # shape_cost = torch.pow(1 - torch.exp(-1 * ar_loss), 4)
    s=  0.5 * (shape_cost) + distance + closs
    return inner_iou,s,g  # IoU


# def wasserstein_loss_improved(pred, target, eps=1e-7, constant=12.8, sigma_center=0.5, sigma_wh=0.3):
#     """
#     Improved Wasserstein Loss with Gaussian kernel weighting on center distance and wh difference.

#     Args:
#         pred (Tensor): Predicted bboxes of format (x_center, y_center, w, h), shape (n, 4).
#         target (Tensor): Corresponding gt bboxes, shape (n, 4).
#         eps (float): Eps to avoid log(0).
#         constant (float): Constant to avoid division by zero.
#         sigma_center (float): Sigma for Gaussian kernel on center distance.
#         sigma_wh (float): Sigma for Gaussian kernel on wh difference.

#     Returns:
#         Tensor: Loss tensor.
#     """
#     # Calculate center point distance
#     center1 = pred[:, :2]
#     center2 = target[:, :2]
#     center_distance = torch.sum((center1 - center2) ** 2, dim=1) + eps

#     # Calculate wh difference
#     w1 = pred[:, 2] + eps
#     h1 = pred[:, 3] + eps
#     w2 = target[:, 2] + eps
#     h2 = target[:, 3] + eps
#     wh_difference = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

#     # Apply Gaussian kernel weighting
#     weight_center = torch.exp(-center_distance / (2 * sigma_center ** 2))
#     weight_wh = torch.exp(-wh_difference / (2 * sigma_wh ** 2))

#     # Calculate weighted center distance and wh difference
#     weighted_center_distance = center_distance * weight_center
#     weighted_wh_difference = wh_difference * weight_wh

#     # Calculate Wasserstein distance with weighted terms
#     wasserstein_2 = weighted_center_distance + weighted_wh_difference

#     # Return the exponential of negative square root of Wasserstein distance divided by constant
#     return torch.exp(-torch.sqrt(wasserstein_2 + eps) / constant)
class LogRelativeLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps  # 防止 log(0)

    def forward(self, pred, target):
        # pred: [N, *], target: [N, *]
         
        ratio = pred / (target + self.eps)
        loss = torch.log(torch.abs(ratio + self.eps - 1) + 1)
        return loss
def wasserstein_loss(pred, target, eps=1e-7, constant=12.8):
    """`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.
    Code is modified from https://github.com/Zzh-tju/CIoU.
    Args:
        pred (Tensor): Predicted bboxes of format (x_center, y_center, w, h),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """

    center1 = pred[:, :2]
    center2 = target[:, :2]

    whs = center1[:, :2] - center2[:, :2]

    center_distance = whs[:, 0] * whs[:, 0] + whs[:, 1] * whs[:, 1] + eps #

    w1 = pred[:, 2]  + eps
    h1 = pred[:, 3]  + eps
    w2 = target[:, 2] + eps
    h2 = target[:, 3] + eps

    wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

    wasserstein_2 = center_distance + wh_distance
    return torch.exp(-torch.sqrt(wasserstein_2) / constant)   
@register()   
class DEIMCriterion(nn.Module):
    """ This class computes the loss for DEIM.
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]     
 
    def __init__(self, \
        matcher,   
        weight_dict, 
        losses,
        alpha=0.2, 
        gamma=2.0,
        num_classes=80,   
        reg_max=32,
        boxes_weight_format=None,    
        share_matched_indices=False, 
        mal_alpha=None,
        use_uni_set=True,
        ):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals.  
            weight_dict: dict containing as key the names of the losses and as values their relative weight.    
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            num_classes: number of object categories, omitting the special no-object category.
            reg_max (int): Max number of the discrete bins in D-FINE.
            boxes_weight_format: format for boxes weight (iou, ).
        """
        super().__init__()
        self.num_classes = num_classes    
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses     
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices     
        self.alpha = alpha   
        self.gamma = gamma   
        self.fgl_targets, self.fgl_targets_dn = None, None 
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max
        self.num_pos, self.num_neg = None, None 
        self.mal_alpha = mal_alpha   
        self.use_uni_set = use_uni_set   
        self.logl1_loss = LogRelativeLoss()
    def loss_labels_focal(self, outputs, targets, indices, num_boxes):   
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) 
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, 
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]   
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes     
 
        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None): 
        assert 'pred_boxes' in outputs  
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]    
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)  
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values
    
        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, 
                                    dtype=torch.int64, device=src_logits.device)   
        target_classes[idx] = target_classes_o 
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]   
  
        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)   
        target_score = target_score_o.unsqueeze(-1) * target  

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score 

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')    
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes  
        return {'loss_vfl': loss}
   
    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx] 
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))    
            ious = torch.diag(ious).detach()
        else:
            ious = values   
   
        src_logits = outputs['pred_logits'] 
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])   
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,    
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)    
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        target_score = target_score.pow(self.gamma) 
        if self.mal_alpha != None:    
            weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
        else:     
            weight = pred_score.pow(self.gamma) * (1 - target) + target
   
        # print(" ### DEIM-gamma{}-alpha{} ### ".format(self.gamma, self.mal_alpha))    
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')  
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes   
        return {'loss_mal': loss}
     
    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]  
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """   
        assert 'pred_boxes' in outputs 
        idx = self._get_src_permutation_idx(indices)  
        src_boxes = outputs['pred_boxes'][idx]   
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
    #     target_boxes_without_normal = target_boxes * 640
    #     targets_w = target_boxes_without_normal[:, 3]
    #     targets_h = target_boxes_without_normal[:, 2]
    # # 计算 batch 中所有边界框的平均宽度和高度和面积
    #     avg_width = targets_w.mean()
    #     avg_height = targets_h.mean()
    #     avg_area= avg_width * avg_height
    #     if avg_area > 9216:
    #         iou_ratio  = 0.7
    #     elif avg_area > 1024:
    #         iou_ratio = 0.5 + (avg_area / 9216) * 0.2
    #     else:
    #     # 动态调整 x，线性插值
    #         iou_ratio =0.1+(avg_area / 1024) * 0.4
        iou_ratio=0.3
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')   
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes   
        nwd = wasserstein_loss(src_boxes, target_boxes)
        nwd_loss = (1.0 - nwd).sum() / num_boxes
        iou,s,g = shape_iou(box1=src_boxes, box2=target_boxes)
        loss_iou = (1.0 - iou).sum() / num_boxes
        giou=g.sum() / num_boxes
        siou=s.sum() / num_boxes
        loss_giou =  (1-iou_ratio)*nwd_loss + (siou+loss_iou)* iou_ratio
        # loss_giou = 1 - torch.diag(generalized_box_iou(\
        #     box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight 
        losses['loss_giou'] = loss_giou
        return losses
  
    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        """Compute Fine-Grained Localization (FGL) Loss
            and Decoupled Distillation Focal (DDF) Loss. """
    
        losses = {}
        if 'pred_corners' in outputs:   
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)     
    
            pred_corners = outputs['pred_corners'][idx].reshape(-1, (self.reg_max+1))  
            ref_points = outputs['ref_points'][idx].detach()   
            with torch.no_grad():    
                if self.fgl_targets_dn is None and 'is_dn' in outputs:
                        self.fgl_targets_dn= bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),    
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])
                if self.fgl_targets is None and 'is_dn' not in outputs:
                        self.fgl_targets = bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])     

            target_corners, weight_right, weight_left = self.fgl_targets_dn if 'is_dn' in outputs else self.fgl_targets     

            ious = torch.diag(box_iou(\
                        box_cxcywh_to_xyxy(outputs['pred_boxes'][idx]), box_cxcywh_to_xyxy(target_boxes))[0])
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()    
  
            losses['loss_fgl'] = self.unimodal_distribution_focal_loss(    
                pred_corners, target_corners, weight_right, weight_left, weight_targets, avg_factor=num_boxes)   

            if 'teacher_corners' in outputs:
                pred_corners = outputs['pred_corners'].reshape(-1, (self.reg_max+1))
                target_corners = outputs['teacher_corners'].reshape(-1, (self.reg_max+1))
                if not torch.equal(pred_corners, target_corners): 
                    weight_targets_local = outputs['teacher_logits'].sigmoid().max(dim=-1)[0]
    
                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool) 
                    mask[idx] = True   
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)
     
                    weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(weight_targets_local.dtype)   
                    weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach() 
 
                    loss_match_local = weight_targets_local * (T ** 2) * (nn.KLDivLoss(reduction='none')
                    (F.log_softmax(pred_corners / T, dim=1), F.softmax(target_corners.detach() / T, dim=1))).sum(-1)
                    if 'is_dn' not in outputs:
                        batch_scale = 8 / outputs['pred_boxes'].shape[0]  # Avoid the influence of batch size per GPU
                        self.num_pos, self.num_neg = (mask.sum() * batch_scale) ** 0.5, ((~mask).sum() * batch_scale) ** 0.5
                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0   
                    loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                    losses['loss_ddf'] = (loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg) / (self.num_pos + self.num_neg)    

        return losses
   
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])   
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx   

    def _get_go_indices(self, indices, indices_aux_list):     
        """Get a matching union set across all decoder layers. """
        results = []
        for indices_aux in indices_aux_list:
            indices = [(torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                        for idx1, idx2 in zip(indices.copy(), indices_aux.copy())]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)     
            count_sort_indices = torch.argsort(counts, descending=True)     
            unique_sorted = unique[count_sort_indices]   
            column_to_row = {}  
            for idx in unique_sorted:     
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)  
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results
     
    def _clear_cache(self):    
        self.fgl_targets, self.fgl_targets_dn = None, None    
        self.own_targets, self.own_targets_dn = None, None   
        self.num_pos, self.num_neg = None, None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):  
        loss_map = {    
            'boxes': self.loss_boxes,   
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,  
            'mal': self.loss_labels_mal,    
            'local': self.loss_local,
        }   
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
   
    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.   
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Retrieve the matching between the outputs of the last layer and the targets  
        indices = self.matcher(outputs_without_aux, targets)['indices']
        self._clear_cache()

        # Get the matching union set across all decoder layers.   
        if 'aux_outputs' in outputs:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            aux_outputs_list = outputs['aux_outputs']  
            if 'pre_outputs' in outputs:    
                aux_outputs_list = outputs['aux_outputs'] + [outputs['pre_outputs']]     
            for i, aux_outputs in enumerate(aux_outputs_list):   
                indices_aux = self.matcher(aux_outputs, targets)['indices'] 
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux) 
            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                indices_enc = self.matcher(aux_outputs, targets)['indices']
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)
 
            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor([num_boxes_go], dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_available_and_initialized():    
                torch.distributed.all_reduce(num_boxes_go)    
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()
        else:
            assert 'aux_outputs' in outputs, ''   
   
        # Compute the average number of target boxes accross all nodes, for normalization purposes   
        num_boxes = sum(len(t["labels"]) for t in targets)    
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)    
        if is_dist_available_and_initialized(): 
            torch.distributed.all_reduce(num_boxes)     
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()    
   
        # Compute all the requested losses, main loss   
        losses = {}    
        for loss in self.losses:    
            # TODO, indices and num_box are different from RT-DETRv2    
            use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
            indices_in = indices_go if use_uni_set else indices
            num_boxes_in = num_boxes_go if use_uni_set else num_boxes     
            meta = self.get_loss_meta_info(loss, outputs, targets, indices_in)
            l_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes_in, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.    
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if 'local' in self.losses:      # only work for local loss     
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']     
                for loss in self.losses:
                    # TODO, indices and num_box are different from RT-DETRv2
                    use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])     
                    indices_in = indices_go if use_uni_set else cached_indices[i] 
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes 
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)
  
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}     
                    losses.update(l_dict)  

        # In case of auxiliary traditional head output at first decoder layer. just for dfine  
        if 'pre_outputs' in outputs:
            aux_outputs = outputs['pre_outputs']
            for loss in self.losses:
                # TODO, indices and num_box are different from RT-DETRv2
                use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                indices_in = indices_go if use_uni_set else cached_indices[-1]    
                num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)     
 
                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict} 
                l_dict = {k + '_pre': v for k, v in l_dict.items()} 
                losses.update(l_dict)     
   
        # In case of encoder auxiliary losses.
        if 'enc_aux_outputs' in outputs:    
            assert 'enc_meta' in outputs, ''    
            class_agnostic = outputs['enc_meta']['class_agnostic']  
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:    
                enc_targets = targets    

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']): 
                for loss in self.losses:  
                    # TODO, indices and num_box are different from RT-DETRv2
                    use_uni_set = self.use_uni_set and (loss == 'boxes')
                    indices_in = indices_go if use_uni_set else cached_indices_enc[i]     
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in) 
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}   
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
     
            if class_agnostic:
                self.num_classes = orig_num_classes

        # In case of cdn auxiliary losses.
        if 'dn_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices_dn = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']
  
            for i, aux_outputs in enumerate(outputs['dn_outputs']):     
                if 'local' in self.losses:      # only work for local loss
                    aux_outputs['is_dn'] = True
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # In case of auxiliary traditional head output at first decoder layer, just for dfine   
            if 'dn_pre_outputs' in outputs:    
                aux_outputs = outputs['dn_pre_outputs']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta) 
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + '_dn_pre': v for k, v in l_dict.items()}
                    losses.update(l_dict)
     
        # For debugging Objects365 pre-train.
        losses = {k:torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}    
        return losses 

    def get_loss_meta_info(self, loss, outputs, targets, indices):     
        if self.boxes_weight_format is None: 
            return {}

        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]  
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(\
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError()
    
        if loss in ('boxes', ):   
            meta = {'boxes_weight': iou}  
        elif loss in ('vfl', 'mal'):
            meta = {'values': iou}
        else:
            meta = {}     
   
        return meta
     
    @staticmethod 
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices
        """     
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device     

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:   
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device) 
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else: 
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))

        return dn_match_indices

 
    def feature_loss_function(self, fea, target_fea):  
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss)     
 

    def unimodal_distribution_focal_loss(self, pred, label, weight_right, weight_left, weight=None, reduction='sum', avg_factor=None):   
        dis_left = label.long()     
        dis_right = dis_left + 1    
  
        loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left.reshape(-1) \
             + F.cross_entropy(pred, dis_right, reduction='none') * weight_right.reshape(-1)   
   
        if weight is not None:     
            weight = weight.float()     
            loss = loss * weight
   
        if avg_factor is not None:
            loss = loss.sum() / avg_factor    
        elif reduction == 'mean':    
            loss = loss.mean()   
        elif reduction == 'sum':
            loss = loss.sum()
     
        return loss

    def get_gradual_steps(self, outputs):   
        num_layers = len(outputs['aux_outputs']) + 1 if 'aux_outputs' in outputs else 1     
        step = .5 / (num_layers - 1)
        opt_list = [.5  + step * i for i in range(num_layers)] if num_layers > 1 else [1]    
        return opt_list
