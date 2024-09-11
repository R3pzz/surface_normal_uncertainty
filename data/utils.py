import random
import numpy as np


def random_crop(img, norm, norm_mask, height, width):
    """randomly crop the input image & surface normal
    """
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y + height, x:x + width, :]
    norm = norm[y:y + height, x:x + width, :]
    norm_mask = norm_mask[y:y + height, x:x + width, :]
    return img, norm, norm_mask


def color_augmentation(image, indoors=True):
    """color augmentation
    """
    # gamma augmentation
    gamma = random.uniform(0.9, 1.1)
    image_aug = image ** gamma

    # brightness augmentation
    if indoors:
        brightness = random.uniform(0.75, 1.25)
    else:
        brightness = random.uniform(0.9, 1.1)
    image_aug = image_aug * brightness

    # color augmentation
    colors = np.random.uniform(0.9, 1.1, size=3)
    white = np.ones((image.shape[0], image.shape[1]))
    color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
    image_aug *= color_image
    image_aug = np.clip(image_aug, 0, 1)
    return image_aug

# samples a noisy normal map on a unit hemisphere facing Z+ with a given mask
# returns: (W, H, 3)
def sample_noisy_normals(W, H) -> np.ndarray:
    n_pix = W*H
    theta = np.random.uniform(0.0, 2.0 * np.pi, n_pix)
    
    # Sample z (cosine of polar angle) from [0, 1) for upper hemisphere
    z = np.random.uniform(0.0, 1.0, n_pix)
    
    # Compute the radius in xy-plane
    r = np.sqrt(1.0 - z**2)
    
    # Compute x and y coordinates using theta
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Stack x, y, z to form 3D normal vectors
    map = np.stack([x, y, z], axis=-1)
    map = np.reshape(map, newshape=(W, H, 3)) # (W, H, 3)
    return map