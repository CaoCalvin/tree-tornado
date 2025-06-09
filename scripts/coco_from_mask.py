# %%
import os
import json
import random # Retained for potential future use, though shuffle isn't strictly needed for single dataset
import shutil
# from PIL import Image, ImageFile # REMOVED
import cv2 # ADDED
import numpy as np # ADDED
from datetime import datetime
from typing import List, Optional # Added for type hinting

import rasterio
from rasterio.transform import Affine
import geopandas
from shapely.geometry import box

# Allow loading of truncated images, can be common with some TIFFs
# ImageFile.LOAD_TRUNCATED_IMAGES = True # REMOVED
# FIX: Increase Pillow's maximum image pixels limit
# Image.MAX_IMAGE_PIXELS = None # REMOVED

def create_coco_dataset_no_arcpy(
    tif_folder_path: str,
    polygon_shapefile_path: str,
    output_directory_path: str,
    scale_factor: float = 1.0,
    target_filenames: Optional[List[str]] = None,
    coco_category_name: str = "tree",
    coco_supercategory_name: str = "natural_object",
    coco_license_name: str = "User Defined",
    coco_license_url: str = "",
    dataset_description: str = "Custom COCO Dataset",
    contributor_name: str = "GIS to COCO Script (No ArcPy)",
    resampling_method = cv2.INTER_AREA # CHANGED
):
    """
    Converts TIF images and polygon annotations (from Shapefile)
    into COCO dataset format without using ArcPy.
    Optionally scales images and annotations.
    Includes detailed debugging print statements for scaling.
    Deletes TIFs from source if no corresponding polygons are found.
    Outputs to a single 'train' set.
    Can filter TIFs to process based on a provided list of filenames.
    Clears and recreates the output directory to ensure a clean structure.
    """
    print(f"Starting COCO dataset creation (No ArcPy) at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  TIF Folder: {tif_folder_path}")
    print(f"  Polygon Shapefile: {polygon_shapefile_path}")
    print(f"  Output Directory: {output_directory_path}")

    # --- MODIFIED SECTION: Clean and prepare output directory ---
    print(f"  Preparing output directory: {output_directory_path}")
    if os.path.exists(output_directory_path):
        print(f"    WARNING: Output directory '{output_directory_path}' already exists.")
        print(f"    It will be completely REMOVED to ensure a clean dataset generation.")
        user_confirmation = input(f"    Type 'YES' to confirm deletion of '{output_directory_path}' and its contents: ")
        if user_confirmation == 'YES':
            try:
                shutil.rmtree(output_directory_path)
                print(f"    Successfully removed existing directory: '{output_directory_path}'")
            except Exception as e:
                print(f"    ERROR: Could not remove existing output directory '{output_directory_path}': {e}")
                print(f"    Please manually clear this directory or check permissions and try again.")
                return None
        else:
            print(f"    Aborted by user. Output directory was not cleared.")
            return None # Or raise an error, or ask for a new path

    try:
        os.makedirs(output_directory_path, exist_ok=False) # Create the root output directory; exist_ok=False to ensure it was indeed removed or didn't exist
        print(f"    Successfully created clean output directory: '{output_directory_path}'")
    except FileExistsError:
        # This case should ideally not be hit if rmtree was successful and no race condition.
        # If it is hit, it implies rmtree didn't fully work or something recreated it.
        print(f"    Output directory '{output_directory_path}' still exists unexpectedly after attempting removal. Please check.")
        return None
    except Exception as e:
        print(f"    ERROR: Could not create output directory '{output_directory_path}': {e}")
        return None
    # --- END OF MODIFIED SECTION ---

    if target_filenames:
        print(f"  Processing only TIFs specified in target_filenames list ({len(target_filenames)} files).")
    if abs(scale_factor - 1.0) > 1e-9 :
        print(f"  Scaling Factor: {scale_factor}")
    else:
        print(f"  Scaling Factor: {scale_factor} (No significant scaling will be applied)")

    images_base_dir = os.path.join(output_directory_path, "images")
    img_output_dir = os.path.join(images_base_dir, "train")
    ann_dir = os.path.join(output_directory_path, "annotations")

    print(f"  Creating necessary sub-directory structure within '{output_directory_path}'...")
    try:
        # os.makedirs(images_base_dir, exist_ok=True) # Redundant if img_output_dir is created
        os.makedirs(img_output_dir, exist_ok=True)   # This will create output_directory_path/images/train
        os.makedirs(ann_dir, exist_ok=True)          # This will create output_directory_path/annotations
        print(f"    Successfully created sub-directory structure.")
        print(f"      Images will be in: {img_output_dir}")
        print(f"      Annotations will be in: {ann_dir}")
    except Exception as e:
        print(f"    ERROR: Could not create sub-directory structure: {e}")
        return None


    base_coco_structure = {
        "info": {
            "description": dataset_description, "version": "1.0", "year": datetime.now().year,
            "contributor": contributor_name, "date_created": datetime.now().isoformat()
        },
        "licenses": [{"id": 1, "name": coco_license_name, "url": coco_license_url}],
        "categories": [{"id": 1, "name": coco_category_name, "supercategory": coco_supercategory_name}]
    }

    try:
        print(f"  Loading polygons from: {polygon_shapefile_path}")
        all_polygons_gdf = geopandas.read_file(polygon_shapefile_path)
        print(f"    Loaded {len(all_polygons_gdf)} polygons.")
        if all_polygons_gdf.crs is None:
            print("    WARNING: Polygon shapefile has no CRS defined. Assuming it matches raster CRS.")
    except Exception as e:
        print(f"    ERROR: Could not read polygon shapefile: {e}")
        return None

    def world_to_pixel_affine(geo_x, geo_y, affine_transform: Affine):
        col, row = ~affine_transform * (geo_x, geo_y)
        return col, row

    all_tif_files_in_folder = [f for f in os.listdir(tif_folder_path) if f.lower().endswith((".tif", ".tiff"))]
    if not all_tif_files_in_folder:
        print("ERROR: No TIF files found in the input folder.")
        return None

    if target_filenames:
        print(f"  Filtering TIFs based on the provided list of {len(target_filenames)} filenames.")
        target_filenames_set = {os.path.basename(f) for f in target_filenames}

        files_in_folder_set = set(all_tif_files_in_folder)
        missing_targets = target_filenames_set - files_in_folder_set
        if missing_targets:
            print(f"      WARNING: The following {len(missing_targets)} target filenames were not found in {tif_folder_path}: {sorted(list(missing_targets))}")

        processed_tif_files = [f for f in all_tif_files_in_folder if os.path.basename(f) in target_filenames_set]

        print(f"      Found {len(processed_tif_files)} matching TIFs to process out of {len(all_tif_files_in_folder)} in the folder.")
        if not processed_tif_files:
            print("ERROR: No TIF files from the target list were found in the input folder. No images to process.")
            # Create empty annotation file and zip as per previous logic if this path is taken.
            # Fall through to the empty dataset handling.
    else:
        processed_tif_files = all_tif_files_in_folder
        print(f"  Processing all {len(processed_tif_files)} TIF files found in the folder.")

    coco_data = json.loads(json.dumps(base_coco_structure))
    coco_data["images"], coco_data["annotations"] = [], []
    global_image_id_counter, global_annotation_id_counter = 0, 0

    print(f"\nProcessing images...")

    for tif_filename in processed_tif_files:
        tif_full_path = os.path.join(tif_folder_path, tif_filename)
        print(f"  Attempting to process TIF: {tif_filename}")
        original_img_width, original_img_height = 0, 0

        try:
            with rasterio.open(tif_full_path) as src_raster:
                original_img_width = src_raster.width
                original_img_height = src_raster.height
                print(f"DEBUG: For {tif_filename} - Rasterio original dimensions: {original_img_width}x{original_img_height}")
                raster_transform = src_raster.transform
                raster_bounds = src_raster.bounds
                raster_crs = src_raster.crs

            polygons_gdf_for_raster = all_polygons_gdf
            if all_polygons_gdf.crs and raster_crs and not all_polygons_gdf.crs.equals(raster_crs):
                print(f"    WARNING: CRS mismatch. Polygon CRS: {all_polygons_gdf.crs}, Raster CRS: {raster_crs}. Attempting to reproject polygons.")
                try:
                    polygons_gdf_for_raster = all_polygons_gdf.to_crs(raster_crs)
                    print("        Polygons reprojected successfully.")
                except Exception as reproj_e:
                    print(f"        ERROR: Failed to reproject polygons for {tif_filename}: {reproj_e}. Skipping this TIF.")
                    continue
            elif all_polygons_gdf.crs is None and raster_crs is not None:
                print(f"    WARNING: Polygons have no CRS. Assuming they match raster CRS for {tif_filename}: {raster_crs}.")

            clip_box_geom = box(raster_bounds.left, raster_bounds.bottom, raster_bounds.right, raster_bounds.top)
            try:
                clip_mask_gdf = geopandas.GeoDataFrame({'geometry': [clip_box_geom]}, crs=raster_crs if raster_crs else polygons_gdf_for_raster.crs)
                clipped_polygons_gdf = geopandas.clip(polygons_gdf_for_raster, clip_mask_gdf)
            except Exception as clip_e:
                print(f"    ERROR during geopandas.clip for {tif_filename}: {clip_e}. Skipping annotations for this TIF.")
                clipped_polygons_gdf = geopandas.GeoDataFrame(columns=['geometry'])

            if clipped_polygons_gdf.empty:
                print(f"    No polygons found within the bounds of {tif_filename} after clipping.")
                print(f"    DELETING TIF file (no corresponding polygons): {tif_full_path}")
                try:
                    os.remove(tif_full_path)
                    print(f"        Successfully deleted {tif_filename}.")
                except OSError as e:
                    print(f"        ERROR: Could not delete {tif_filename}: {e}")
                continue
            else:
                print(f"    Found {len(clipped_polygons_gdf)} polygons after clipping for {tif_filename}.")

            global_image_id_counter += 1
            current_image_id = global_image_id_counter
            print(f"    Processing TIF: {tif_filename} for dataset (Image ID: {current_image_id})")

            scaled_img_width = int(original_img_width * scale_factor)
            scaled_img_height = int(original_img_height * scale_factor)
            print(f"DEBUG: For {tif_filename} - Calculated scaled dimensions: {scaled_img_width}x{scaled_img_height} (using scale_factor: {scale_factor})")

            coco_data["images"].append({
                "id": current_image_id, "file_name": tif_filename,
                "width": scaled_img_width, "height": scaled_img_height,
                "license": 1, "date_captured": ""
            })

            for _, poly_row in clipped_polygons_gdf.iterrows():
                polygon_geom_shapely = poly_row.geometry
                map_unit_area = polygon_geom_shapely.area
                if polygon_geom_shapely.is_empty or not polygon_geom_shapely.geom_type in ['Polygon', 'MultiPolygon']:
                    continue
                global_annotation_id_counter += 1
                coco_segments_for_ann = []
                all_scaled_pixel_coords_x, all_scaled_pixel_coords_y = [], []
                geoms_to_process = []
                if polygon_geom_shapely.geom_type == 'Polygon':
                    geoms_to_process.append(polygon_geom_shapely)
                elif polygon_geom_shapely.geom_type == 'MultiPolygon':
                    geoms_to_process.extend(list(polygon_geom_shapely.geoms))

                for single_poly_shapely in geoms_to_process:
                    exterior_coords = list(single_poly_shapely.exterior.coords)
                    current_part_scaled_pixels_ext = []
                    for coord_tuple in exterior_coords:
                        geo_x, geo_y = coord_tuple[0], coord_tuple[1]
                        original_pixel_x, original_pixel_y = world_to_pixel_affine(geo_x, geo_y, raster_transform)
                        scaled_pixel_x = original_pixel_x * scale_factor
                        scaled_pixel_y = original_pixel_y * scale_factor
                        clamped_scaled_x = max(0.0, min(scaled_pixel_x, float(scaled_img_width)))
                        clamped_scaled_y = max(0.0, min(scaled_pixel_y, float(scaled_img_height)))
                        current_part_scaled_pixels_ext.extend([clamped_scaled_x, clamped_scaled_y])
                        all_scaled_pixel_coords_x.append(clamped_scaled_x)
                        all_scaled_pixel_coords_y.append(clamped_scaled_y)
                    if current_part_scaled_pixels_ext:
                        coco_segments_for_ann.append(current_part_scaled_pixels_ext)

                    for interior_ring in single_poly_shapely.interiors:
                        interior_coords = list(interior_ring.coords)
                        current_part_scaled_pixels_int = []
                        for coord_tuple in interior_coords:
                            geo_x, geo_y = coord_tuple[0], coord_tuple[1]
                            original_pixel_x, original_pixel_y = world_to_pixel_affine(geo_x, geo_y, raster_transform)
                            scaled_pixel_x = original_pixel_x * scale_factor
                            scaled_pixel_y = original_pixel_y * scale_factor
                            clamped_scaled_x = max(0.0, min(scaled_pixel_x, float(scaled_img_width)))
                            clamped_scaled_y = max(0.0, min(scaled_pixel_y, float(scaled_img_height)))
                            current_part_scaled_pixels_int.extend([clamped_scaled_x, clamped_scaled_y])
                            all_scaled_pixel_coords_x.append(clamped_scaled_x)
                            all_scaled_pixel_coords_y.append(clamped_scaled_y)
                        if current_part_scaled_pixels_int:
                            coco_segments_for_ann.append(current_part_scaled_pixels_int)

                if not coco_segments_for_ann or not all_scaled_pixel_coords_x: continue
                min_x_s = min(all_scaled_pixel_coords_x)
                min_y_s = min(all_scaled_pixel_coords_y)
                max_x_s = max(all_scaled_pixel_coords_x)
                max_y_s = max(all_scaled_pixel_coords_y)
                bbox_coco_scaled = [min_x_s, min_y_s, max_x_s - min_x_s, max_y_s - min_y_s]
                pixel_area_on_map_sq = abs(raster_transform.a * raster_transform.e)
                original_pixel_area = (map_unit_area / pixel_area_on_map_sq) if pixel_area_on_map_sq > 1e-9 else 0
                scaled_pixel_area = original_pixel_area * (scale_factor ** 2)
                coco_data["annotations"].append({
                    "id": global_annotation_id_counter, "image_id": current_image_id, "category_id": 1,
                    "segmentation": coco_segments_for_ann, "area": scaled_pixel_area,
                    "bbox": bbox_coco_scaled, "iscrowd": 0
                })

            # --- MODIFIED IMAGE HANDLING SECTION ---
            cv2_img_original = cv2.imread(tif_full_path, cv2.IMREAD_UNCHANGED)
            if cv2_img_original is None:
                 print(f"    ERROR: CV2 could not open or read {tif_filename}. Skipping image save. JSON entry will remain but image file will be missing.")
                 continue # Skip to the next TIF file

            h, w = cv2_img_original.shape[:2]
            print(f"DEBUG: For {tif_filename} - CV2 opened original dimensions: {w}x{h}")
            destination_image_path = os.path.join(img_output_dir, tif_filename)

            if abs(scale_factor - 1.0) < 1e-9:
                print(f"DEBUG: For {tif_filename} - Copying image, no scaling. Path: {destination_image_path}")
                shutil.copy2(tif_full_path, destination_image_path)
            else:
                print(f"DEBUG: For {tif_filename} - Entering scaling block. Target: {scaled_img_width}x{scaled_img_height}. Resampling: cv2.INTER_AREA")
                scaled_img_cv2 = cv2.resize(cv2_img_original, (scaled_img_width, scaled_img_height), interpolation=resampling_method)
                sh, sw = scaled_img_cv2.shape[:2]
                print(f"DEBUG: For {tif_filename} - CV2 resized dimensions (before save): {sw}x{sh}")
                try:
                    print(f"DEBUG: For {tif_filename} - Attempting to save SCALED image using CV2 to {destination_image_path}")
                    success = cv2.imwrite(destination_image_path, scaled_img_cv2)
                    if success:
                        print(f"DEBUG: For {tif_filename} - Successfully saved SCALED image using CV2 to {destination_image_path}")
                    else:
                        print(f"    ERROR: Could not save scaled {tif_filename} using CV2 (imwrite returned False).")
                except Exception as save_e:
                    print(f"    ERROR: Could not save {tif_filename} using CV2 after scaling: {save_e}.")
            # No close needed for cv2.imread
            # --- END OF MODIFIED IMAGE HANDLING SECTION ---

        except rasterio.errors.RasterioIOError as rio_e:
            print(f"    ERROR: Rasterio could not open or read {tif_filename}: {rio_e}")
        except FileNotFoundError:
            print(f"    ERROR: File not found for processing (was it moved/deleted?): {tif_full_path}")
        except Exception as e:
            print(f"    ERROR processing file {tif_filename}: {e.__class__.__name__} - {e}")
            import traceback
            traceback.print_exc()

    if not coco_data["images"]:
        print("\nWARNING: No images were processed and added to the dataset. This could be due to:")
        print("  - All TIFs lacked corresponding polygons and were deleted.")
        print("  - The target_filenames list (if provided) resulted in no processable TIFs.")
        print("  - Other errors during processing of all TIFs.")
        print("An empty 'train.json' will be created, and the ZIP archive may be empty or incomplete.")

    output_json_filename = os.path.join(ann_dir, "instances_train.json")
    with open(output_json_filename, 'w') as f: json.dump(coco_data, f, indent=4)
    print(f"\n  Saved annotations to {output_json_filename}")

    parent_dir_of_output = os.path.dirname(output_directory_path.rstrip(os.sep))
    if not parent_dir_of_output:
        parent_dir_of_output = "."
    archive_base_name = os.path.join(parent_dir_of_output, "archive")

    print(f"\nCreating ZIP archive from contents of: {output_directory_path}")
    print(f"Archive will be saved as: {archive_base_name}.zip")

    try:
        zip_output_path = shutil.make_archive(
            base_name=archive_base_name,
            format='zip',
            root_dir=output_directory_path
        )
        print(f"Successfully created ZIP archive: {zip_output_path}")
        return zip_output_path
    except Exception as e:
        print(f"ERROR creating ZIP archive: {e}")
        import traceback
        traceback.print_exc()
        # If zipping fails but data was generated, output_directory_path still contains the data
        print(f"Data was generated in '{output_directory_path}' but zipping failed.")
        return None # Or return output_directory_path to indicate partial success
    finally:
        print(f"COCO dataset creation finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    # --- USER CONFIGURATION ---
    input_tifs_folder = r"C:\Users\kevin\dev\tornado-tree-destruction-ef\tornado-tree-destruction-ef\data\2022_woodstock\TIFFS"
    input_polygon_shp = r"C:\Users\kevin\dev\tornado-tree-destruction-ef\tornado-tree-destruction-ef\data\2022_woodstock\Woodstock Lake ArcGIS merged final.shp"
    output_coco_dir = r"C:\Users\kevin\dev\tornado-tree-destruction-ef\tornado-tree-destruction-ef\data\2022_woodstock\coco"  # MODIFY THIS

    image_annotation_scale_factor = .5

    category = "fallen" # Example: more specific category
    super_category = "tree" # Example
    description = "Dataset of tree damage from drone imagery for COCO."
    contributor = "Automated Script"
    license_nm = "CC0 - Public Domain"
    license_link = "https://creativecommons.org/publicdomain/zero/1.0/"

    # target_tifs = ["DJI_0001.TIF", "DJI_0002.TIF"]
    target_tifs = None

    try:
        generated_zip_file = create_coco_dataset_no_arcpy(
            tif_folder_path=input_tifs_folder,
            polygon_shapefile_path=input_polygon_shp,
            output_directory_path=output_coco_dir,
            scale_factor=image_annotation_scale_factor,
            target_filenames=target_tifs,
            coco_category_name=category,
            coco_supercategory_name=super_category,
            dataset_description=description,
            contributor_name=contributor,
            coco_license_name=license_nm,
            coco_license_url=license_link
            # No need to pass resampling_method, it uses the new cv2 default
        )

        if generated_zip_file:
            print(f"\n--- SCRIPT FINISHED SUCCESSFULLY ---")
            print(f"COCO dataset ZIP archive created at: {generated_zip_file}")
            print(f"The contents of the ZIP (images/train/ and annotations/train.json) are from: {output_coco_dir}")
        else:
            print(f"\n--- SCRIPT FINISHED WITH ERRORS OR NO IMAGES PROCESSED ---")
            print(f"COCO dataset generation may have failed or produced an empty/incomplete dataset. Please review the logs.")
            if os.path.exists(output_coco_dir) and any(os.scandir(output_coco_dir)):
                 print(f"Intermediate files might be present in: {output_coco_dir}")


    except ImportError as ie:
        print(f"An import error occurred: {ie}")
        print("Please ensure you have installed all required libraries:")
        print("pip install rasterio geopandas shapely opencv-python pyproj") # MODIFIED
    except Exception as e:
        print(f"An unexpected error occurred in the main execution block: {e}")
        import traceback
        traceback.print_exc()


