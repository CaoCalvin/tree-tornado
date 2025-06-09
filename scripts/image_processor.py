# %% [markdown]
# # Setup

# %%
import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
import csv
from datetime import datetime, timezone
from pathlib import Path
import shutil

# --- Configuration ---
BASE_OUTPUT_DIR = "..\dataset_processed" # Base output directory for processed images and masks
IMAGES_OUT_DIR = Path(BASE_OUTPUT_DIR, "images")
MASKS_OUT_DIR = Path(BASE_OUTPUT_DIR, "masks")
LAST_MODIFIED_FILE = Path(BASE_OUTPUT_DIR, "lastmodified.txt")

# User-specified path to the root of local image folders
LOCAL_IMAGE_ROOT_DIR = r"..\dataset\images" #Needs to be changed by user

# Default downscaling interpolation method
DOWNSCALE_INTERPOLATION = cv2.INTER_AREA

# Grayscale mapping for masks
LABEL_TO_VALUE = {
    "upright": 0,
    "fallen": 1,
    "other": 2,
    "unlabeled": 3, # Used for checking, but chips with "unlabeled" are skipped
    "incomplete": 4 # Used for checking, images with "incomplete" are skipped
}
NO_LABEL_TEMP_VALUE = 255 # Temporary value for pixels not yet labeled in a chip mask

# Chip settings
CHIP_SIZE = 512
STRIDE = 256

# %% [markdown]
# # Define helper functions

# %%

def parse_cvat_xml(xml_file):
    """Parses the CVAT XML file."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_file}: {e}")
        return None

    project_meta = root.find('meta')
    project_updated_str = project_meta.find('project/updated').text if project_meta.find('project/updated') is not None else None
    
    tasks_data = {}
    raw_tasks = project_meta.findall('.//task')
    for task_elem in raw_tasks:
        task_id = task_elem.find('id').text
        task_name = task_elem.find('name').text
        task_updated_str = task_elem.find('updated').text
        try:
            task_updated_dt = datetime.fromisoformat(task_updated_str)
        except ValueError:
            # Try to parse if it has milliseconds with 6 digits and a Z
            if '.' in task_updated_str and task_updated_str.endswith('Z'):
                 task_updated_str = task_updated_str.split('.')[0] + '.' + task_updated_str.split('.')[1][:6] + '+00:00'
                 task_updated_dt = datetime.fromisoformat(task_updated_str)
            elif '.' in task_updated_str and task_updated_str.endswith('+00:00'): # Python < 3.11 needs Z converted
                task_updated_dt = datetime.fromisoformat(task_updated_str.replace('Z', '+00:00'))

            else: # Fallback if parsing still fails, or provide a default
                print(f"Warning: Could not parse timestamp '{task_updated_str}' for task {task_id}. Using current time as fallback.")
                task_updated_dt = datetime.now(timezone.utc)


        tasks_data[task_id] = {
            "name": task_name,
            "updated": task_updated_dt,
            "images": []
        }

    images_annotations = {}
    for image_elem in root.findall(".//image"):
        image_id = image_elem.get("id")
        image_name = image_elem.get("name")
        task_id = image_elem.get("task_id")
        width = int(image_elem.get("width"))
        height = int(image_elem.get("height"))

        annotations = []
        for poly_elem in image_elem.findall("polygon"):
            label = poly_elem.get("label")
            points_str = poly_elem.get("points")
            points = [list(map(float, p.split(','))) for p in points_str.split(';')]
            annotations.append({"type": "polygon", "label": label, "points": points, "source_elem": poly_elem})

        for box_elem in image_elem.findall("box"):
            label = box_elem.get("label")
            xtl = float(box_elem.get("xtl"))
            ytl = float(box_elem.get("ytl"))
            xbr = float(box_elem.get("xbr"))
            ybr = float(box_elem.get("ybr"))
            points = [(xtl, ytl), (xbr, ytl), (xbr, ybr), (xtl, ybr)]
            annotations.append({"type": "box", "label": label, "points": points, "source_elem": box_elem})
        
        if task_id in tasks_data:
            tasks_data[task_id]["images"].append({
                "id": image_id,
                "name": image_name,
                "width": width,
                "height": height,
                "annotations": annotations
            })
        else:
            print(f"Warning: Image {image_name} references task_id {task_id} which was not found in project metadata.")

    # Filter out tasks that have no images associated directly in the parsed structure
    # (though typically all images should link to a task)
    # Also, combine task info with image annotations
    processed_tasks = {}
    for task_id, task_info in tasks_data.items():
        if task_info["images"]:
             processed_tasks[task_id] = task_info


    return processed_tasks


def read_server_paths(csv_path):
    """Reads server paths from the CSV file."""
    paths = {}
    try:
        with open(csv_path, mode='r', newline='') as infile:
            reader = csv.reader(infile)
            next(reader)  # Skip header
            for row in reader:
                if row: # ensure row is not empty
                    location_name, server_folder_path, *_ = row # handles rows with more than 2 columns
                    paths[location_name.strip()] = server_folder_path.strip()
    except FileNotFoundError:
        print(f"Warning: Server paths CSV '{csv_path}' not found. Image download will not be possible.")
    except Exception as e:
        print(f"Error reading server paths CSV '{csv_path}': {e}")
    return paths 

def get_last_modified_timestamp():
    """Reads the last modified timestamp from the file."""
    if LAST_MODIFIED_FILE.exists():
        try:
            return datetime.fromisoformat(LAST_MODIFIED_FILE.read_text().strip())
        except ValueError:
            print(f"Warning: Could not parse timestamp from {LAST_MODIFIED_FILE}. Processing all images.")
            return datetime.min.replace(tzinfo=timezone.utc) # Process all if timestamp is invalid
    return datetime.min.replace(tzinfo=timezone.utc) # Process all if file doesn't exist

def set_last_modified_timestamp():
    """Writes the current timestamp to the last modified file."""
    if not BASE_OUTPUT_DIR.exists():
        os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    LAST_MODIFIED_FILE.write_text(datetime.now(timezone.utc).isoformat())

def scale_annotations(annotations, scale_factor):
    """Scales annotation coordinates."""
    scaled_annotations = []
    for ann in annotations:
        scaled_points = [[p[0] * scale_factor, p[1] * scale_factor] for p in ann["points"]]
        scaled_annotations.append({**ann, "points": scaled_points})
    return scaled_annotations

def ensure_dir(path): 
    """Ensures a directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)

# Updated get_image_full_path function
def get_image_full_path(image_name, task_name, server_paths_csv_data):
    """Tries to find the image locally, then on the server.
    Downloads the image locally if found on the server.
    Returns the local path if found, None otherwise.
    """
    local_path = Path(LOCAL_IMAGE_ROOT_DIR, task_name, image_name)
    if local_path.exists():
        print(f"Found image locally: {local_path}")
        return str(local_path)
    
    print(f"Image not found locally: {local_path}. Attempting server lookup...")
    if task_name in server_paths_csv_data:
        server_folder_path = server_paths_csv_data[task_name]
        # Check if server_folder_path is a UNC path or local-like
        if server_folder_path.startswith("\\\\") or Path(server_folder_path).drive:
            potential_server_path = Path(server_folder_path, image_name)
            if potential_server_path.exists():
                print(f"Found image on server: {potential_server_path}")
                # Ensure the local folder exists before downloading
                if not local_path.parent.exists():
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    # "Download" the image by copying it from the server to the local destination
                    shutil.copy2(potential_server_path, local_path)
                    print(f"Image downloaded to {local_path}")
                    return str(local_path)
                except Exception as e:
                    print(f"Error downloading image from server: {e}")
                    return None
            else:
                print(f"Image {image_name} not found at server path: {potential_server_path}")
        else:
            print(f"Server path for {task_name} ('{server_folder_path}') is not a recognized file system path.")
    else:
        print(f"No server path entry found for location: {task_name} in CSV.")
    
    print(f"Failed to find image '{image_name}' for task '{task_name}' both locally and on server.")
    return None


def polygon_intersects_bbox(poly_points, bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y):
    """
    Basic check if a polygon's bounding box intersects with the given bounding box.
    This is an approximation; for precise checks, more complex geometry operations are needed.
    """
    if not poly_points:
        return False
    
    poly_min_x = min(p[0] for p in poly_points)
    poly_max_x = max(p[0] for p in poly_points)
    poly_min_y = min(p[1] for p in poly_points)
    poly_max_y = max(p[1] for p in poly_points)

    # Check for non-overlap
    if poly_max_x < bbox_min_x or poly_min_x > bbox_max_x:
        return False
    if poly_max_y < bbox_min_y or poly_min_y > bbox_max_y:
        return False
    return True


# --- Main Processing Logic ---
def process_images(cvat_xml_path, server_paths_csv_path):
    """Main function to process images and generate dataset."""
    
    print(f"Starting dataset generation process at {datetime.now()}")
    print(f"CVAT XML: {cvat_xml_path}")
    print(f"Server Paths CSV: {server_paths_csv_path}")
    print(f"Local Image Root: {LOCAL_IMAGE_ROOT_DIR}")
    print(f"Output Directory: {BASE_OUTPUT_DIR}")

    tasks = parse_cvat_xml(cvat_xml_path)
    if not tasks:
        print("No tasks found or error parsing XML. Exiting.")
        return

    server_paths_data = read_server_paths(server_paths_csv_path)
    last_processed_time = get_last_modified_timestamp()
    print(f"Last processed time: {last_processed_time}")
    
    ensure_dir(IMAGES_OUT_DIR)
    ensure_dir(MASKS_OUT_DIR)

    images_processed_count = 0
    chips_generated_count = 0

    for task_id, task_data in tasks.items():
        task_name = task_data["name"]
        task_updated_time = task_data["updated"]

        # Ensure both datetimes are timezone-aware (UTC)
        if task_updated_time.tzinfo is None:
            task_updated_time = task_updated_time.replace(tzinfo=timezone.utc)
        if last_processed_time.tzinfo is None:
            last_processed_time = last_processed_time.replace(tzinfo=timezone.utc)

        if task_updated_time <= last_processed_time:
            print(f"Skipping task '{task_name}' (ID: {task_id}) as it has not been updated since last run ({task_updated_time}).")
            continue
        
        print(f"\nProcessing task: {task_name} (ID: {task_id}, Updated: {task_updated_time})")

        task_images_out_dir = IMAGES_OUT_DIR / task_name
        task_masks_out_dir = MASKS_OUT_DIR / task_name
        ensure_dir(task_images_out_dir)
        ensure_dir(task_masks_out_dir)

        for image_info in task_data["images"]:
            image_name = image_info["name"]
            print(f"  Processing image: {image_name}")

            current_annotations = image_info["annotations"]

            # Error Check 1 for entire satellite image
            all_labels_in_image = [ann["label"] for ann in current_annotations]
            has_incomplete = "incomplete" in all_labels_in_image
            
            # Check if image has annotations and if they are all "unlabeled"
            if not all_labels_in_image: # No annotations at all
                is_only_unlabeled_or_empty = True
            else:
                is_only_unlabeled_or_empty = all(label == "unlabeled" for label in all_labels_in_image)

            if has_incomplete:
                print(f"    ERROR: Image '{image_name}' contains 'incomplete' labels. Skipping this image.")
                continue 
            if is_only_unlabeled_or_empty:
                 print(f"    ERROR: Image '{image_name}' has no valid labels or only 'unlabeled' labels. Skipping this image.")
                 continue


            full_image_path = get_image_full_path(image_name, task_name, server_paths_data)
            if not full_image_path:
                print(f"    Could not find image '{image_name}'. Skipping.")
                continue

            try:
                # Read with alpha channel if present, otherwise discard it later if not needed.
                img = cv2.imread(full_image_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"    Failed to load image: {full_image_path}. Skipping.")
                    continue
            except Exception as e:
                print(f"    Exception loading image {full_image_path}: {e}. Skipping.")
                continue
            
            # If image has 4 channels (e.g., RGBA), convert to BGR for processing
            if img.ndim == 3 and img.shape[2] == 4:
                print(f"    Image {image_name} has 4 channels. Converting to BGR.")
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif img.ndim == 2: # Grayscale image
                print(f"    Image {image_name} is grayscale. Converting to BGR.")
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.ndim == 3 and img.shape[2] != 3:
                 print(f"    Image {image_name} has an unsupported number of channels: {img.shape[2]}. Skipping.")
                 continue


            h, w = img.shape[:2]
            annotations_to_use = current_annotations
            scale_factor = 1.0

            if h > 10000 or w > 10000:
                print(f"    Image dimensions ({w}x{h}) > 10000. Downscaling by factor of 2.")
                scale_factor = 0.5
                new_w, new_h = int(w * scale_factor), int(h * scale_factor)
                img = cv2.resize(img, (new_w, new_h), interpolation=DOWNSCALE_INTERPOLATION)
                # print(f"(1) Image size tested: {w}x{h}, downscaled to {img.shape[1]}x{img.shape[0]}")
                # Overwrite the original image with the downscaled version
                cv2.imwrite(full_image_path, img)
                print(f"    Overwrote original image with downscaled version at: {full_image_path}")
                annotations_to_use = scale_annotations(current_annotations, scale_factor)
                h, w = new_h, new_w # Update dimensions
                print(f"    New dimensions: {w}x{h}")


            for y in range(0, h - CHIP_SIZE + 1, STRIDE):
                for x in range(0, w - CHIP_SIZE + 1, STRIDE):
                    # print(f"img size inside loop: {img.shape[1]}x{img.shape[0]}, chip position: ({x},{y})")
                    img_chip = img[y:y + CHIP_SIZE, x:x + CHIP_SIZE]
                    mask_chip = np.full((CHIP_SIZE, CHIP_SIZE), NO_LABEL_TEMP_VALUE, dtype=np.uint8)

                    chip_intersecting_polygons_info = []
                    chip_has_any_actual_label = False
                    chip_contains_unlabeled_label_type = False
                    
                    # Gather polygons relevant to this chip
                    for ann in annotations_to_use:
                        # Basic bounding box intersection test for speed
                        if not polygon_intersects_bbox(ann["points"], x, y, x + CHIP_SIZE, y + CHIP_SIZE):
                            continue

                        if ann["label"] == "unlabeled":
                            chip_contains_unlabeled_label_type = True
                            break 
                        
                        # Convert points to be relative to the chip's origin
                        relative_points = [[p[0] - x, p[1] - y] for p in ann["points"]]
                        
                        if ann["label"] in LABEL_TO_VALUE:
                            chip_intersecting_polygons_info.append({
                                "points": np.array(relative_points, dtype=np.int32),
                                "label": ann["label"],
                                "value": LABEL_TO_VALUE[ann["label"]]
                            })
                            if ann["label"] != "unlabeled": # "unlabeled" itself is not an "actual" data label for training
                                chip_has_any_actual_label = True
                        # else: # Should not happen if XML is well-formed and labels are in LABEL_TO_VALUE
                           # print(f"    Warning: Unknown label '{ann['label']}' in image {image_name}. Skipping this annotation for chip.")


                    if chip_contains_unlabeled_label_type:
                        # print(f"    Skipping chip ({x},{y}) for {image_name}: contains 'unlabeled' type polygon.")
                        continue
                    
                    if not chip_has_any_actual_label:
                        # print(f"    Skipping chip ({x},{y}) for {image_name}: no actual labels found.")
                        continue

                    # --- Special Upright Rule Check ---
                    all_polys_in_chip_are_upright = True
                    if not chip_intersecting_polygons_info: # Should be caught by chip_has_any_actual_label
                        all_polys_in_chip_are_upright = False
                    for poly_info in chip_intersecting_polygons_info:
                        if poly_info["label"] != "upright":
                            all_polys_in_chip_are_upright = False
                            break
                    
                    apply_all_upright_rule = False
                    if all_polys_in_chip_are_upright and chip_intersecting_polygons_info:
                        quadrant_hits = [False, False, False, False] # TL, TR, BL, BR
                        q_size = CHIP_SIZE // 2
                        
                        for poly_info in chip_intersecting_polygons_info: # These are all "upright"
                            # Create a temporary mask for this single polygon to check quadrant presence
                            temp_poly_mask = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=np.uint8)
                            cv2.fillPoly(temp_poly_mask, [poly_info["points"]], 1)
                            
                            if not quadrant_hits[0] and np.any(temp_poly_mask[0:q_size, 0:q_size]):
                                quadrant_hits[0] = True
                            if not quadrant_hits[1] and np.any(temp_poly_mask[0:q_size, q_size:CHIP_SIZE]):
                                quadrant_hits[1] = True
                            if not quadrant_hits[2] and np.any(temp_poly_mask[q_size:CHIP_SIZE, 0:q_size]):
                                quadrant_hits[2] = True
                            if not quadrant_hits[3] and np.any(temp_poly_mask[q_size:CHIP_SIZE, q_size:CHIP_SIZE]):
                                quadrant_hits[3] = True
                        
                        if all(quadrant_hits):
                            apply_all_upright_rule = True

                    if apply_all_upright_rule:
                        mask_chip[:, :] = LABEL_TO_VALUE["upright"]
                        # print(f"    Applied all-upright rule to chip ({x},{y}) for {image_name}.")
                    else:
                        # --- Standard Mask Drawing ---
                        # Sort polygons? (e.g. by area, or a predefined label priority if needed)
                        # For now, draw in order of appearance.
                        for poly_info in chip_intersecting_polygons_info:
                             if poly_info["label"] in ["upright", "fallen", "other"]: # Only draw trainable labels
                                cv2.fillPoly(mask_chip, [poly_info["points"]], poly_info["value"])
                        
                        # --- Fill remaining unlabelled areas (if any) with 'upright' ---
                        mask_chip[mask_chip == NO_LABEL_TEMP_VALUE] = LABEL_TO_VALUE["upright"]

                    # Save chip
                    chip_filename_base = f"{Path(image_name).stem}_chip_{int(x/256)}_{int(y/256)}"
                    img_chip_path = task_images_out_dir / f"{chip_filename_base}.png"
                    mask_chip_path = task_masks_out_dir / f"{chip_filename_base}_mask.png"
                    
                    cv2.imwrite(str(img_chip_path), img_chip)
                    cv2.imwrite(str(mask_chip_path), mask_chip)
                    chips_generated_count += 1
            
            images_processed_count +=1

    if images_processed_count > 0:
        set_last_modified_timestamp()
        print(f"\nFinished processing. {images_processed_count} images processed, {chips_generated_count} chips generated.")
        print(f"Timestamp updated in {LAST_MODIFIED_FILE}")
    else:
        print("\nNo images needed processing based on timestamps.")


# %% [markdown]
# # Run the processing function

# %%

if __name__ == "__main__":
    # --- User Input ---
    # Ensure these paths are correct
    cvat_xml_file = r"..\resources\annotations.xml"  # Path to your CVAT XML file
    server_paths_file = r"..\resources\serverpaths.csv" # Path to your CSV file with server paths
    
    # Update LOCAL_IMAGE_ROOT_DIR at the top of the script if your local dataset root is different.

    if not Path(cvat_xml_file).exists():
        print(f"Error: CVAT XML file not found at '{cvat_xml_file}'")
    elif not Path(server_paths_file).exists() and LOCAL_IMAGE_ROOT_DIR == "": # Only critical if local dir not set or for fallback
         print(f"Warning: Server paths CSV not found at '{server_paths_file}'. Image download/fallback may not work.")
         # Decide if you want to proceed or exit if CSV is mandatory for some workflows.
         # For now, it will proceed but log warnings if images aren't found locally.
    else:
        process_images(cvat_xml_file, server_paths_file)

# %% [markdown]
# # Display Generated Tiles

# %%
import os
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def hex_to_rgb(hex_color):
    """Converts a hex color string to an (R, G, B) tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Define the color mapping for mask values
# Greyscale class values: 0 = upright, 1 = fallen, 2 = other, 3 = unlabelled
COLOR_MAP_HEX = {
    0: "#b7f2a6",  # upright (greenish)
    1: "#c71933",  # fallen (reddish)
    2: "#ffcc33",  # other (yellowish)
    3: "#2a23bf"   # unlabelled (bluish)
}

COLOR_MAP_RGB = {
    value: hex_to_rgb(hex_code) for value, hex_code in COLOR_MAP_HEX.items()
}

# Alpha value for the mask overlay (0-255, where 255 is fully opaque)
# 128 is about 50% transparency
MASK_ALPHA = 128

def display_chip_grid(dataset_root, collection_name, filename_prefix, 
                      x_start, y_start, x_end, y_end, chip_size_render=(256, 256)):
    """
    Displays a grid of image chips with their masks overlaid.

    Args:
        dataset_root (str): Path to the root directory of the dataset 
                            (e.g., "./dataset").
        collection_name (str): Name of the subfolder within 'images' and 'masks'
                               (e.g., "centreglassville", "crozier").
        filename_prefix (str): The prefix for the image and mask files
                               (e.g., "centreglassville"). This is the {filename}
                               part in {filename}_{xidx}_{yidx}.png.
        x_start (int): The starting x-index of the chip rectangle.
        y_start (int): The starting y-index of the chip rectangle.
        x_end (int): The ending x-index of the chip rectangle (inclusive).
        y_end (int): The ending y-index of the chip rectangle (inclusive).
        chip_size_render (tuple): Tuple (width, height) to resize images for display if needed.
                                  Original image size is used by default for masks processing.
    """
    
    num_cols = x_end - x_start + 1
    num_rows = y_end - y_start + 1
    
    if num_cols <= 0 or num_rows <= 0:
        print("Error: Invalid start/end coordinates. End coordinates must be greater than or equal to start.")
        return

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    
    # If only one chip, axes might not be an array, so make it one
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = np.array([axes])
    elif num_cols == 1:
        axes = np.array([[ax] for ax in axes])


    for i in range(num_rows):
        for j in range(num_cols):
            chip_y = y_start + i
            chip_x = x_start + j
            
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal') # Ensure chips are not distorted

            img_filename = f"{filename_prefix}_{chip_x}_{chip_y}.png"
            mask_filename = f"{filename_prefix}_{chip_x}_{chip_y}_mask.png"

            img_path = os.path.join(dataset_root, "images", collection_name, img_filename)
            mask_path = os.path.join(dataset_root, "masks", collection_name, mask_filename)
            
            try:
                # Load original image
                img_pil = Image.open(img_path).convert("RGB")
                
                # Load grayscale mask
                mask_pil_gray = Image.open(mask_path).convert("L") # Ensure it's grayscale
                
                # Ensure mask and image are the same size (use original mask size for processing)
                if img_pil.size != mask_pil_gray.size:
                    print(f"Warning: Image {img_filename} and mask {mask_filename} have different sizes. "
                          f"Image: {img_pil.size}, Mask: {mask_pil_gray.size}. Resizing mask to image size for overlay.")
                    mask_pil_gray = mask_pil_gray.resize(img_pil.size, Image.NEAREST)

                mask_np_gray = np.array(mask_pil_gray)

                # Create an RGBA overlay mask
                # Initialize with fully transparent
                overlay_rgba = np.zeros((mask_np_gray.shape[0], mask_np_gray.shape[1], 4), dtype=np.uint8)
                
                for val, rgb_color in COLOR_MAP_RGB.items():
                    pixels_to_color = (mask_np_gray == val)
                    overlay_rgba[pixels_to_color, 0] = rgb_color[0]
                    overlay_rgba[pixels_to_color, 1] = rgb_color[1]
                    overlay_rgba[pixels_to_color, 2] = rgb_color[2]
                    overlay_rgba[pixels_to_color, 3] = MASK_ALPHA 
                
                overlay_pil = Image.fromarray(overlay_rgba, "RGBA")

                # Resize image for display if chip_size_render is specified
                display_img_pil = img_pil.resize(chip_size_render, Image.Resampling.LANCZOS)
                display_overlay_pil = overlay_pil.resize(chip_size_render, Image.Resampling.NEAREST)

                # Display the image
                ax.imshow(display_img_pil)
                
                # Overlay the colored mask
                ax.imshow(display_overlay_pil) # Alpha is handled by the RGBA image
                ax.set_title(f"({chip_x},{chip_y})", fontsize=8)

            except FileNotFoundError:
                # If image or mask is not found, display a blank space
                ax.set_facecolor('lightgray')
                ax.text(0.5, 0.5, f"No Chip\n({chip_x},{chip_y})", 
                        horizontalalignment='center', verticalalignment='center', 
                        fontsize=8, transform=ax.transAxes)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
            except Exception as e:
                ax.set_facecolor('pink') # Error indicator
                ax.text(0.5, 0.5, f"Error\n({chip_x},{chip_y})\n{type(e).__name__}",
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=6, transform=ax.transAxes, color='red')
                print(f"Error processing chip ({chip_x},{chip_y}): {e}")


    plt.suptitle(f"Chip Grid: {collection_name} ({filename_prefix}) | X: {x_start}-{x_end}, Y: {y_start}-{y_end}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.show()

if __name__ == '__main__':
    # --- Configuration ---
    # IMPORTANT: Replace this with the actual path to your dataset's root folder
    DATASET_ROOT_PATH = "../dataset_processed" 
    
    # Name of the subfolder (e.g., "centreglassville" or "crozier")
    # This is the name of the folder inside 'images' and 'masks'
    COLLECTION_NAME = "centreglassville" 
    # COLLECTION_NAME = "crozier"

    # This is the {filename} part of your files, e.g., "centreglassville" if files are 
    # "centreglassville_0_0.png"
    FILENAME_PREFIX = "Glassville_ortho_5cm16_chip" 
    # FILENAME_PREFIX = "22_Crozier_461000_5385000_chip"

    # Define the rectangle of chips to display (inclusive indices)
    TOP_LEFT_X = 0
    TOP_LEFT_Y = 0
    BOTTOM_RIGHT_X = 20 
    BOTTOM_RIGHT_Y = 20

    # (Optional) Define a size for rendering each chip in the plot
    # Set to None to use original image size (can lead to very large plots)
    # Good values might be (128,128) or (256,256) for viewing many chips
    CHIP_DISPLAY_SIZE = (128, 128) 

    # Call the main function to display the grid
    display_chip_grid(
        dataset_root=DATASET_ROOT_PATH,
        collection_name=COLLECTION_NAME,
        filename_prefix=FILENAME_PREFIX,
        x_start=TOP_LEFT_X,
        y_start=TOP_LEFT_Y,
        x_end=BOTTOM_RIGHT_X,
        y_end=BOTTOM_RIGHT_Y,
        chip_size_render=CHIP_DISPLAY_SIZE
    )


