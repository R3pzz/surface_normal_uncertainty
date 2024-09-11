# Preprocessor - takes in rgb and gt normals
# Augmentations:
#   - hflip -- not used as the foot image layout is always vertical
#   - random crop -- we may want to use this one, still have to check the synfoot images
#   - color augmentation -- useless shit i guess, although have to experiment on it
# Final step - conversion to pytorch from numpy

import os
import numpy as np
import random
from PIL import Image

import torch
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .utils import sample_noisy_normals

MAX_IMAGES = 1000

class SynFootLoader(object):
    def __init__(self, args, fldr_path):
        self.training_samples = SynFootTrainPreprocessor(args, fldr_path)
        #self.testing_samples = SynFootTestPreprocessor(args, fldr_path)
        self.testing_samples = SynFootTrainPreprocessor(args, fldr_path) # same preprocessor as we're not modifying the original images

        # multigpu not supported as this code is intended to be ran in colab on an a100 gpu
        self.train_data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   drop_last=True)
        self.test_data = DataLoader(self.testing_samples, 1,
                               shuffle=False,
                               num_workers=1,
                               pin_memory=False)

# we'd want to preprocess the data the same way as for the NYU dataset to support uncertainty-guided sampling
class SynFootTrainPreprocessor(Dataset):
    def __init__(self, args, fldr_path):
        super(SynFootTrainPreprocessor, self).__init__()

        # problem here is, synfoot is 
        self.fldr_path = fldr_path
        self.filenames = os.listdir(os.path.join(fldr_path, 'rgb'))[:MAX_IMAGES]

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.input_height = args.input_height
        self.input_width = args.input_width

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        sample_path = self.filenames[idx]

        # img path and norm path
        rgb_path = os.path.join(self.fldr_path, 'rgb', sample_path)
        norm_path = os.path.join(self.fldr_path, 'normals', sample_path)
        img_name = sample_path.rstrip('.png')

        # read img / normal
        rgb_image = Image.open(rgb_path).convert("RGB").resize(
            size=(self.input_width, self.input_height), 
            resample=Image.BILINEAR
        )
        norm_image = Image.open(norm_path).convert("RGB").resize(
            size=(self.input_width, self.input_height), 
            resample=Image.NEAREST
        )

        # to numpy
        img = np.array(rgb_image).astype(np.float32) / 255.0 # (H, W, 3)
        norm_gt = np.array(norm_image).astype(np.uint8) # (H, W, 3)

        # a mask for VALID normals
        norm_valid_mask = np.logical_not(
            np.logical_and(
                np.logical_and(
                    norm_gt[:, :, 0] == 0, norm_gt[:, :, 1] == 0),
                norm_gt[:, :, 2] == 0))

        # sample a noisy normal map
        norm_noise = sample_noisy_normals(
            self.input_width,
            self.input_height
        ).astype(np.float32) # (H, W, 3)

        # fill in INVALID pixels with noise
        norm_gt = norm_gt.astype(np.float32) / 255.0
        norm_gt[~norm_valid_mask, :] = norm_noise[~norm_valid_mask, :]
        norm_gt = norm_gt * 2.0 - 1.0

        # add a channel dimension
        norm_valid_mask = norm_valid_mask[:, :, np.newaxis] # (H, W, 1)

        # to tensors
        img = self.normalize(torch.from_numpy(img).permute(2, 0, 1))            # (3, H, W)
        norm_gt = torch.from_numpy(norm_gt).permute(2, 0, 1)                    # (3, H, W)
        norm_valid_mask = torch.from_numpy(norm_valid_mask).permute(2, 0, 1)    # (1, H, W)

        sample = {'img': img,
                  'norm': norm_gt,
                  'norm_valid_mask': norm_valid_mask,
                  'scene_name': 'train',
                  'img_name': img_name}

        return sample