import torch
import torch.nn.functional as F

def UG_angular_vMF_loss(pred_norm, pred_kappa, gt_norm):
  # pred_norm: (B*3*W*H)
  # gt_norm:(B*3*W*H)
  # pred_kappa: (B*W*H)

  # compute the angular similarity between gt and prediction
  dot = torch.cosine_similarity(pred_norm, gt_norm)  # (B*H*W)
  dot = torch.clamp(dot, min=-1.0, max=1.0)

  # compute the resulting angle
  ang = torch.acos(dot)  # (B*H*W)

  # compute the losses (angle and concentration)
  kappa_loss = torch.log((1 + torch.exp(-pred_kappa * torch.pi)) / (pred_kappa**2 + 1))
  ang_loss = pred_kappa * ang

  return torch.mean(kappa_loss) + torch.mean(ang_loss)

def pixelwise_loss(pred_list, coord_list, gt_norm, gt_mask):
  loss = 0.0
  for (pred, coord) in zip(pred_list, coord_list):
    if coord is None:
      pred = F.interpolate(pred, size=[gt_norm.size(2), gt_norm.size(3)], mode='bilinear', align_corners=True)
      pred_norm, pred_kappa = pred[:, :3, :, :], pred[:, 3:, :, :]

      # mask out the background pixels
      fg_norm = pred_norm[gt_mask.expand(-1, 3, -1, -1)]
      fg_kappa = pred_kappa[gt_mask]

      # calculate the foreground loss
      fg_loss = UG_angular_vMF_loss(fg_norm, fg_kappa, gt_norm[gt_mask.expand(-1, 3, -1, -1)])

      # mask out the foreground pixels
      mask_inv = torch.logical_not(gt_mask)
      bg_norm = pred_norm[mask_inv.expand(-1, 3, -1, -1)]
      bg_kappa = pred_kappa[mask_inv]

      # detach the background normals to not fuck up our predictions
      bg_norm = bg_norm.detach()

      # calculate the foreground loss
      bg_loss = UG_angular_vMF_loss(bg_norm, bg_kappa, gt_norm[mask_inv.expand(-1, 3, -1, -1)])
      
      loss = loss + fg_loss + bg_loss * 0.1

    else:
      sampled_gt_norm = F.grid_sample(gt_norm, coord, mode='nearest', align_corners=True)  # (B, 3, 1, N)
      sampled_gt_norm = sampled_gt_norm[:, :, 0, :]  # (B, 3, N)
      
      sampled_gt_mask = F.grid_sample(gt_mask.float(), coord, mode='nearest', align_corners=True)  # (B, 1, 1, N)
      sampled_gt_mask = sampled_gt_mask[:, :, 0, :] > 0.5  # (B, 1, N)

      pred_norm, pred_kappa = pred[:, :3, :], pred[:, 3:, :]

      # mask out the background pixels
      fg_norm = pred_norm[sampled_gt_mask.expand(-1, 3, -1)]
      fg_kappa = pred_kappa[sampled_gt_mask]

      # calculate the foreground loss
      fg_loss = UG_angular_vMF_loss(fg_norm, fg_kappa, sampled_gt_norm[sampled_gt_mask.expand(-1, 3, -1)])

      # mask out the foreground pixels
      mask_inv = torch.logical_not(sampled_gt_mask)
      bg_norm = pred_norm[mask_inv.expand(-1, 3, -1)]
      bg_kappa = pred_kappa[mask_inv]

      # detach the background normals to not fuck up our predictions
      bg_norm = bg_norm.detach()

      # calculate the foreground loss
      bg_loss = UG_angular_vMF_loss(bg_norm, bg_kappa, sampled_gt_norm[mask_inv.expand(-1, 3, -1)])
      
      loss = loss + fg_loss + bg_loss * 0.1

  return loss