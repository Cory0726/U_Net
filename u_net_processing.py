import cv2
import os
from pathlib import Path

def binarize_images(input_dir, output_dir=None, threshold=127):
    """
    Convert all images in a folder to binary (0 or 255) based on a threshold.

    Args:
        input_dir (str): Path to the input folder containing images.
        output_dir (str): Path to the output folder for saving binary images.
                          Defaults to 'input_dir/binarized'.
        threshold (int): Threshold value for binarization (default: 127).
    """
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir / "binarized"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Supported image extensions
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    for img_path in input_dir.iterdir():
        if img_path.suffix.lower() not in exts:
            continue

        # Read image as grayscale
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Cannot read image: {img_path}")
            continue

        # Apply binary thresholding
        _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

        # Save result
        save_path = output_dir / img_path.name
        cv2.imwrite(str(save_path), binary)
        print(f"Converted: {img_path.name} → {save_path}")

    print("All images have been binarized!")

# Example usage
if __name__ == "__main__":
    binarize_images("images")  # Replace "images" with your input folder path
