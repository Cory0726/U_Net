import argparse
import logging
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance

from utils.data_loading import BasicDataset
from unet import UNet

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
        img,
        model,
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
def main():
    # Load image
    img = Image.open('test_img/M1_01_intensity_grayscale.png')

    # Final mask
    final_mask_np = None

    # Adjust the brightness of the image 50 - 95 %
    brightness_levels = [i/100 for i in range(50, 96, 5)]
    print(f'brightness_levels: {brightness_levels}')

    for b in brightness_levels:
        print(f'Processing brightness: {b:.2f}')

        enhancer = ImageEnhance.Brightness(img)
        img_bright = enhancer.enhance(b)

        # Predict the mask by U-Net model
        result_mask = run_predict(
            img=img_bright,
            model='trained_weight/Hand_Seg_EGTEA_plus_S640480G_Scale05_Score08994_20251123.pth',
            scale=0.5,
            num_of_channels=1,
            num_of_classes=2,
            mask_threshold=0.5,
            bilinear=False,
        )

        # Save images and masks at different brightness levels
        output_img_path = f'image_process_temp/img_brightness_{int(b*100):3d}.png'
        output_mask_path = f'image_process_temp/mask_brightness_{int(b*100):3d}.png'
        img_bright.save(output_img_path)
        print(f'Saved: {output_img_path}')
        result_mask.save(output_mask_path)
        print(f'Saved: {output_mask_path}')

        # Mask convert to numpy type
        mask_np = np.array(result_mask)
        # Pixel-wise OR merging
        if final_mask_np is None:
            final_mask_np = mask_np.copy()
        else:
            final_mask_np = np.maximum(final_mask_np, mask_np)
    # Save the final mask
    final_mask = Image.fromarray(final_mask_np)
    final_mask.save(f'result_mask/final_mask.png')

if __name__ == '__main__':
    img = Image.open('test_img/M1_01_intensity_grayscale.png')
    img_np = np.array(img)
    print(img_np.shape, img_np.dtype, img_np.max(), img_np.min())

    main()
