import cv2
import os
import glob
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

def dataset_mask_filter(image_dir, mask_dir, mask_ratio_range=(0.01, 0.9)):
    """
    Delete image + mask pairs based on the foreground ratio of binary masks (0/255).

    Args:
        image_dir (str): Directory containing original images.
        mask_dir (str): Directory containing mask images. Mask filenames must end with "_mask".
        mask_ratio_range (tuple): (min_ratio, max_ratio). Samples outside this range will be deleted.

    Returns:
        dict: Summary statistics.
    """
    min_ratio, max_ratio = mask_ratio_range

    mask_paths = glob.glob(os.path.join(mask_dir, "*_mask.*"))

    total_masks = 0
    removed = 0
    kept = 0

    for mask_path in mask_paths:
        total_masks += 1

        # Read mask (binary 0/255)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[WARN] Failed to read mask: {mask_path}")
            continue

        total_pixels = mask.size
        fg_pixels = np.count_nonzero(mask == 255)
        ratio = fg_pixels / float(total_pixels)

        # Check if outside allowed ratio range
        if ratio < min_ratio or ratio > max_ratio:
            removed += 1

            # Determine corresponding image filename
            dirname, filename = os.path.split(mask_path)
            stem, ext = os.path.splitext(filename)

            if not stem.endswith("_mask"):
                print(f"[WARN] Mask filename does not end with '_mask': {filename}")
                continue

            image_stem = stem[:-5]  # Remove "_mask"
            image_path = os.path.join(image_dir, image_stem + ext)

            print(f"[DELETE] Mask foreground ratio = {ratio:.4f} (outside {min_ratio} ~ {max_ratio})")
            print(f"         Deleting mask:  {mask_path}")
            print(f"         Deleting image: {image_path}")

            # Delete mask
            try:
                os.remove(mask_path)
            except Exception as e:
                print(f"[ERROR] Failed to delete mask: {e}")

            # Delete corresponding image (if exists)
            if os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except Exception as e:
                    print(f"[ERROR] Failed to delete image: {e}")
        else:
            kept += 1

    summary = {
        "total_masks": total_masks,
        "kept": kept,
        "removed": removed,
    }

    print("\n=== Dataset Cleaning Finished ===")
    print(f"Total mask files: {total_masks}")
    print(f"Kept:             {kept}")
    print(f"Removed:          {removed}")

    return summary

def batch_convert_jpg_to_png(src_dir: str, dst_dir: str):
    """
    Batch convert all .jpg files in a folder to .png format.

    Args:
        src_dir (str): Source directory containing .jpg files.
        dst_dir (str): Destination directory to save .png files.

    Notes:
        - The function reads each JPG file using OpenCV and writes it as a PNG file.
        - File extensions are handled automatically; only the format changes.
        - The original JPG files are NOT deleted.
    """

    # Create destination folder if it doesn't exist
    os.makedirs(dst_dir, exist_ok=True)

    # Find all .jpg files inside the directory
    jpg_files = glob.glob(os.path.join(src_dir, "*.jpg"))

    if len(jpg_files) == 0:
        print("[INFO] No JPG files found in:", src_dir)
        return

    for jpg_path in jpg_files:
        # Extract filename and remove extension
        filename = os.path.basename(jpg_path)
        stem = os.path.splitext(filename)[0]

        # Create PNG output path
        png_path = os.path.join(dst_dir, stem + ".png")

        # Read the image
        img = cv2.imread(jpg_path)
        if img is None:
            print(f"[WARN] Could not read file: {jpg_path}")
            continue

        # Write image as PNG
        cv2.imwrite(png_path, img)

        print(f"[CONVERTED] {jpg_path} → {png_path}")

    print("\n=== Conversion Completed ===")
    print(f"Source folder:      {src_dir}")
    print(f"Output folder:      {dst_dir}")

def convert_folder_to_grayscale(input_dir, output_dir):
    """
    Convert all images inside a folder to grayscale and save them to another folder.

    Args:
        input_dir (str): Path to the input folder containing images.
        output_dir (str): Path to the output folder for storing converted grayscale images.

    Notes:
        - Supported formats: .jpg, .jpeg, .png, .bmp, .tiff
        - Output image keeps original filename.
    """

    # Create output folder if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Supported image extensions
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    # Loop through every file in the input folder
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(valid_ext):
            # Full path to input image
            img_path = os.path.join(input_dir, filename)

            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"[Warning] Failed to read: {filename}")
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Full path for output image
            out_path = os.path.join(output_dir, filename)

            # Save grayscale image
            cv2.imwrite(out_path, gray)

    print(f"Conversion complete. Grayscale images saved to: {output_dir}")


# Main
if __name__ == "__main__":
    convert_folder_to_grayscale(
        input_dir='C:/Users/User/PycharmProjects/Dataset/Process_EGTEA_Gaze_plus_640_480_MaskFilter_05_40_GrayScale/origin_imgs',
        output_dir='C:/Users/User/PycharmProjects/Dataset/Process_EGTEA_Gaze_plus_640_480_MaskFilter_05_40_GrayScale/imgs'
    )