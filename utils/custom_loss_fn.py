import torch
import torch.nn.functional as F

def vMF_masked_loss(pred, gt_norm, gt_mask):
  gt_mask = gt_mask[:, 0, :, :] # (B, H, W)
  
  # extract normals and concentration
  norm = pred[:, :3, :, :] # (B, 3, H, W)
  kappa = pred[:, 3:, :, :] # (B, H, W)
  
  # split into bg and foot
  kappa_foot = kappa[gt_mask]
  kappa_bg = kappa[~gt_mask]
  kappa_bg = kappa_bg.detach()
  
  # calculate the similarity for both bg and foot
  dot = torch.cosine_similarity(norm, gt_norm)
  dot_foot = dot[gt_mask]
  dot_bg = dot[~gt_mask]

  loss_foot = torch.mean(
    kappa_foot * torch.acos(dot_foot) \
      + torch.log((1 + torch.exp(-kappa_foot * torch.pi)) / (kappa_foot**2 + 1))
  )
  
  loss_bg = torch.mean(
    kappa_bg * torch.acos(dot_bg) \
      + torch.log((1 + torch.exp(-kappa_bg * torch.pi)) / (kappa_bg**2 + 1))
  )
  
  return loss_foot + loss_bg * 0.1

def pixelwise_loss(pred_list, coord_list, gt_norm, gt_mask):
  loss = 0.0
  for (pred, coord) in zip(pred_list, coord_list):
    if coord is None:
      pred = F.interpolate(pred, size=[gt_norm.size(2), gt_norm.size(3)], mode='bilinear', align_corners=True)
      loss += vMF_masked_loss(pred, gt_norm, gt_mask)

    else:
      sampled_gt_norm = F.grid_sample(gt_norm, coord, mode='nearest', align_corners=True)  # (B, 3, 1, N)
      #sampled_gt_norm = sampled_gt_norm[:, :, 0, :]  # (B, 3, N)
      
      sampled_gt_mask = F.grid_sample(gt_mask.float(), coord, mode='nearest', align_corners=True)  # (B, 1, 1, N)
      #sampled_gt_mask = sampled_gt_mask[:, :, 0, :] > 0.5  # (B, 1, N)
      sampled_gt_mask = sampled_gt_mask > 0.5  # (B, 1, 1, N)
      
      pred = pred.unsqueeze(-2)

      loss += vMF_masked_loss(pred, sampled_gt_norm, sampled_gt_mask)

  return loss