import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()  # Set model to evaluation mode
    # Preprocess input image: resize, normalize, convert to numpy array
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    # Add batch dimension (1, C, H, W)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()  # Forward pass, move output to CPU

        # Resize prediction to original image size
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')

        # For multi-class segmentation:
        # Each pixel has n class scores, take  the index (class ID) with the highest score per pixel.
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            # For binary segmentation:
            # Apply sigmoid to convert logits to probabilities (0-1 range),
            # then threshold to produce a boolean mask (foreground/background)
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()

def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)

def run_predict(
        input_img_file,
        model = 'trained_weight/Hand_Seg_EGTEA_plus_S640480G_Scale05_Score08994_20251123.pth',
        scale=0.5,
        num_of_channels = 1,
        num_of_classes = 2,
        mask_threshold = 0.5,
        bilinear = False,
):

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    net = UNet(n_channels=num_of_channels, n_classes=num_of_classes, bilinear=bilinear)  # n_channel=3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    logging.info(f'Predicting image ...')

    mask = predict_img(net=net,
                        full_img=img,
                        scale_factor=scale,
                        out_threshold=mask_threshold,
                        device=device)
    # Return mask
    return mask_to_image(mask, mask_values)



if __name__ == '__main__':
    run_predict()
