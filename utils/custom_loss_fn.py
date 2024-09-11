import torch
import torch.nn.functional as F

def vMF(pred_norm, pred_kappa, gt_norm, gt_norm_mask):
  dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1)

  valid_mask = gt_norm_mask[:, 0, :, :].float() \
              * (dot.detach() < 0.999).float() \
              * (dot.detach() > -0.999).float()
  valid_mask = valid_mask > 0.5

  dot_foot = dot[valid_mask]
  dot_bg = dot[~valid_mask]
  
  kappa_foot = pred_kappa[:, 0, :, :][valid_mask]
  kappa_bg = pred_kappa[:, 0, :, :][~valid_mask]
  kappa_bg.detach()

  loss_foot = - torch.log(kappa_foot) \
                   - (kappa_foot * (dot_foot - 1)) \
                   + torch.log(1 - torch.exp(- 2 * kappa_foot))

  loss_bg = - torch.log(kappa_bg) \
                   - (kappa_bg * (dot_bg - 1)) \
                   + torch.log(1 - torch.exp(- 2 * kappa_bg))
  return torch.mean(loss_foot) + torch.mean(loss_bg) * 0.1

def pixelwise_loss(pred_list, coord_list, gt_norm, gt_mask):
  loss = 0.0
  for (pred, coord) in zip(pred_list, coord_list):
    if coord is None:
      pred = F.interpolate(pred, size=[gt_norm.size(2), gt_norm.size(3)], mode='bilinear', align_corners=True)
      pred_norm, pred_kappa = pred[:, :3, :, :], pred[:, 3:, :, :]

      loss += vMF(pred, gt_norm, gt_mask)

    else:
      sampled_gt_norm = F.grid_sample(gt_norm, coord, mode='nearest', align_corners=True)  # (B, 3, 1, N)
      #sampled_gt_norm = sampled_gt_norm[:, :, 0, :]  # (B, 3, N)
      
      sampled_gt_mask = F.grid_sample(gt_mask.float(), coord, mode='nearest', align_corners=True)  # (B, 1, 1, N)
      #sampled_gt_mask = sampled_gt_mask[:, :, 0, :] > 0.5  # (B, 1, N)
      sampled_gt_mask = sampled_gt_mask > 0.5  # (B, 1, 1, N)
      
      pred = pred.unsqueeze(-2)
      pred_norm, pred_kappa = pred[:, :3, :, :], pred[:, 3:, :, :]

      loss += vMF(pred, sampled_gt_norm, sampled_gt_mask)

  return loss