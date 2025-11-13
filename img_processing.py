import cv2
import os
import shutil
import numpy as np
from pathlib import Path
from PIL import Image

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

def rename_and_move_files(input_dir, output_dir, base_name="img", start_num=1, suffix=""):
    """
    Rename and move all image files from the input directory to the output directory.
    After moving, the original files in the input directory will be deleted.

    New filenames will follow the pattern: base_name_0001<suffix>.<extension>

    Args:
        input_dir (str): Folder containing the original image files.
        output_dir (str): Destination folder for renamed images.
        base_name (str): Base name (prefix) for renamed files. Default is 'image'.
        start_num (int): Starting number for the new filenames. Default is 1.
        suffix (str): String appended after the index, before file extension.
                        e.g., "_mask", "_label". Default is "".
    """

    # --- Check if input directory exists ---
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # --- Create the output directory if it doesn’t exist ---
    os.makedirs(output_dir, exist_ok=True)

    # --- Collect all image files (you can add more extensions if needed) ---
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(exts)])

    if not files:
        print("No image files found in the input directory.")
        return

    # --- Start renaming and moving ---
    count = start_num
    for f in files:
        src = os.path.join(input_dir, f)
        ext = os.path.splitext(f)[1]              # Get file extension
        new_name = f"{base_name}_{count:05d}{suffix}{ext}" # Format: image_00001.jpg
        dst = os.path.join(output_dir, new_name)

        # Move the file (this automatically deletes it from input_dir)
        shutil.move(src, dst)
        # Pring a message after each move
        print(f" Moved: {f} to {new_name}")
        count += 1

    print(f" Renamed and moved {count - start_num} files to: {output_dir}")

def resize_and_convert_image(input_folder, output_folder, target_size):
    """
    Resize normal images and convert to PNG (RGB).
    Args:
        input_folder (str): Path to input images.
        output_folder (str): Path to save PNG output.
        target_size (tuple): (width, height)
    """
    os.makedirs(output_folder, exist_ok=True)
    valid_ext = ('.jpg', '.jpeg', '.png', '.gif', '.tif', '.tiff')

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(valid_ext):
            continue

        in_path = os.path.join(input_folder, filename)
        out_name = os.path.splitext(filename)[0] + ".png"
        out_path = os.path.join(output_folder, out_name)

        try:
            with Image.open(in_path) as img:

                # If GIF: take first frame
                if getattr(img, "is_animated", False):
                    img.seek(0)

                # Convert to RGB
                img = img.convert("RGB")

                # Resize with bilinear interpolation (smooth)
                img = img.resize(target_size, Image.BILINEAR)

                # Save as PNG
                img.save(out_path, format="PNG")
                print(f"[IMAGE] {filename} → {out_path}")

        except Exception as e:
            print(f"[IMAGE] Failed {filename}: {e}")

def resize_and_convert_mask(input_folder, output_folder, target_size):
    """
    Resize segmentation masks and convert to PNG.
    Args:
        input_folder (str): Path to original masks.
        output_folder (str): Path to save resized PNG masks.
        target_size (tuple): (width, height)
    """
    os.makedirs(output_folder, exist_ok=True)
    valid_ext = ('.jpg', '.jpeg', '.png', '.gif', '.tif', '.tiff')

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(valid_ext):
            continue

        in_path = os.path.join(input_folder, filename)
        out_name = os.path.splitext(filename)[0] + ".png"
        out_path = os.path.join(output_folder, out_name)

        try:
            with Image.open(in_path) as img:

                # Handle animated images (GIF)
                if getattr(img, "is_animated", False):
                    img.seek(0)

                # Convert to grayscale to unify mode
                if img.mode not in ["L", "I", "F"]:
                    img = img.convert("L")

                # Resize with NEAREST to preserve discrete mask values
                img = img.resize(target_size, Image.NEAREST)

                # Save without any binarization or value change
                img.save(out_path, format="PNG")
                print(f"[MASK] {filename} → {out_path}")

        except Exception as e:
            print(f"[MASK] Failed {filename}: {e}")

# Main
if __name__ == "__main__":
    # Process imgs
    rename_and_move_files(
        input_dir="C:/Users/lkfu5/PycharmProjects/Temp/Images",
        output_dir="C:/Users/lkfu5/PycharmProjects/Temp/imgs_renamed",
        base_name="img",
        start_num=0
    )
    resize_and_convert_image(
        input_folder="C:/Users/lkfu5/PycharmProjects/Temp/imgs_renamed",
        output_folder="C:/Users/lkfu5/PycharmProjects/Temp/imgs_renamed_resized",
        target_size=(640, 480)
    )
    # Process masks
    rename_and_move_files(
        input_dir="C:/Users/lkfu5/PycharmProjects/Temp/Masks",
        output_dir="C:/Users/lkfu5/PycharmProjects/Temp/masks_renamed",
        base_name="img_mask",
        start_num=0,
        suffix="_mask"
    )
    binarize_images(
        input_dir="C:/Users/lkfu5/PycharmProjects/Temp/masks_renamed",
        output_dir="C:/Users/lkfu5/PycharmProjects/Temp/masks_renamed_binarized",
        threshold=127
    )
    resize_and_convert_mask(
        input_folder="C:/Users/lkfu5/PycharmProjects/Temp/masks_renamed_binarized",
        output_folder="C:/Users/lkfu5/PycharmProjects/Temp/masks_renamed_binarized_resized",
        target_size=(640, 480)
    )
