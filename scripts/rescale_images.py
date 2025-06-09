# %%
import os
import cv2
import sys

def rescale_image(image_path: str, scale: float) -> None:
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return

    height, width = image.shape[:2]
    new_size = (int(width * scale), int(height * scale))

    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    # Overwrite original image
    success = cv2.imwrite(image_path, resized)
    if not success:
        print(f"Failed to write resized image: {image_path}")

def rescale_all_tifs(folder_path: str, scale: float) -> None:
    if not os.path.isdir(folder_path):
        raise ValueError(f"Provided path is not a directory: {folder_path}")
    if scale <= 0:
        raise ValueError("Scaling factor must be positive.")

    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(('.tif', '.tiff')):
                full_path = os.path.join(root, filename)
                print(f"Rescaling: {full_path}")
                rescale_image(full_path, scale)

if __name__ == "__main__":
    folder = r"C:\Users\kevin\dev\tornado-tree-destruction-ef\dataset\images\ayr"
    try:
        scale_factor = 0.5
    except ValueError:
        print("Scaling factor must be a number.")
        sys.exit(1)

    rescale_all_tifs(folder, scale_factor)



