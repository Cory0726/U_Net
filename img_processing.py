import cv2
import os
import shutil
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

def rename_and_move_files(input_dir, output_dir, base_name="img", start_num=1):
    """
    Rename and move all image files from the input directory to the output directory.
    After moving, the original files in the input directory will be deleted.

    New filenames will follow the pattern: base_name_0001.<extension>

    Args:
        input_dir (str): Folder containing the original image files.
        output_dir (str): Destination folder for renamed images.
        base_name (str): Base name (prefix) for renamed files. Default is 'image'.
        start_num (int): Starting number for the new filenames. Default is 1.
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
        new_name = f"{base_name}_{count:05d}{ext}" # Format: image_00001.jpg
        dst = os.path.join(output_dir, new_name)

        # Move the file (this automatically deletes it from input_dir)
        shutil.move(src, dst)
        # Pring a message after each move
        print(f" Moved: {f} to {new_name}")
        count += 1

    print(f" Renamed and moved {count - start_num} files to: {output_dir}")

# Main
if __name__ == "__main__":
    binarize_images("images")
