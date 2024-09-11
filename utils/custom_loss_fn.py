import torch
import torch.nn.functional as F

def vMF_masked_loss(pred, gt_norm, gt_mask):
  # extract normals and concentration
  norm = pred[:, :3, :, :] # (B, 3, H, W)
  kappa = pred[:, 3:, :, :] # (B, H, W)
  kappa.unsqueeze(1) # (B, 1, H, W)

  # expand the mask so it covers all 3 channels
  gt_mask_exp = gt_mask.expand(-1, 3, -1, -1) # (B, 3, H, W)
  
  # split bg and foot predictions
  bg_norm = norm[~gt_mask_exp] # (N)
  foot_norm = norm[gt_mask_exp] # (N)
  
  bg_kappa = kappa[~gt_mask] # (N)
  foot_kappa = kappa[gt_mask] # (N)

  # detach kappa for bg so we don't optimize over shit pixels
  bg_kappa = bg_kappa.detach()

  # split bg and foot gt normals
  gt_bg_norm = gt_norm[~gt_mask_exp] # (N) # random noise sampled during the data processing stage
  gt_foot_norm = gt_norm[gt_mask_exp] # (N) # ground truth foot normal map

  # loss fn:
  # foot_kappa * acos(foot_norm * gt_foot_norm) + log((1 + exp(-foot_kappa * PI)) / (1 + square(foot_kappa)))

  foot_dev = torch.clamp(foot_norm * gt_foot_norm, min=-1.0, max=1.0) # (N)
  foot_loss = torch.mean(
    foot_kappa * torch.acos(foot_dev) \
      + torch.log((1 + torch.exp(-foot_kappa * torch.pi)) / (foot_kappa**2 + 1))
  )

  bg_dev = torch.clamp(bg_norm * gt_bg_norm, min=-1.0, max=1.0) # (N)
  bg_loss = torch.mean(
    bg_kappa * torch.acos(bg_dev) \
      + torch.log((1 + torch.exp(-bg_kappa * torch.pi)) / (bg_kappa**2 + 1))
  )
  
  return foot_loss + bg_loss * .1

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