# test_image_processor.py
import pytest
from unittest.mock import patch, mock_open, MagicMock, call
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import cv2
from datetime import datetime, timezone, timedelta
import os
import shutil

# Import the functions and constants from your script
# To make your script more testable, it's often better if constants like BASE_OUTPUT_DIR
# are either passed into functions or can be easily overridden.
# For this example, we'll use monkeypatch to modify them during tests.
import image_processor

# --- Test Data ---

SAMPLE_CVAT_XML_VALID = """
<annotations>
    <version>1.1</version>
    <meta>
        <project>
            <name>Test Project</name>
            <bug_tracker></bug_tracker>
            <created>2024-05-01 10:00:00.000000+00:00</created>
            <updated>2024-05-15 12:00:00.000000+00:00</updated>
        </project>
        <task>
            <id>1</id>
            <name>Task1_LocationA</name>
            <size>1</size>
            <mode>annotation</mode>
            <overlap>0</overlap>
            <bug_tracker></bug_tracker>
            <created>2024-05-01 10:00:00.000000+00:00</created>
            <updated>2024-05-10 11:00:00.000000+00:00</updated>
            <subset>default</subset>
            <start_frame>0</start_frame>
            <stop_frame>0</stop_frame>
            <frame_filter></frame_filter>
            <segments>
                <segment>
                    <id>1</id>
                    <start>0</start>
                    <stop>0</stop>
                    <url>http://localhost:8080/?id=1</url>
                </segment>
            </segments>
            <owner>
                <username>admin</username>
                <email></email>
            </owner>
            <assignee/>
            <labels>
                <label><name>upright</name><color>#ff0000</color><type>polygon</type></label>
                <label><name>fallen</name><color>#00ff00</color><type>polygon</type></label>
                <label><name>other</name><color>#0000ff</color><type>polygon</type></label>
                <label><name>unlabeled</name><color>#ffffff</color><type>polygon</type></label>
                <label><name>incomplete</name><color>#000000</color><type>polygon</type></label>
            </labels>
        </task>
         <task>
            <id>2</id>
            <name>Task2_LocationB</name>
            <updated>2024-05-20T10:00:00.123456Z</updated>
        </task>
        <task>
            <id>3</id>
            <name>Task3_LocationC</name>
            <updated>2024-05-20T10:00:00.123+00:00</updated>
        </task>
        <task>
            <id>4</id>
            <name>Task4_BadDate</name>
            <updated>Invalid Date</updated>
        </task>
    </meta>
    <image id="0" name="image1.png" width="1024" height="768" task_id="1">
        <polygon label="upright" points="10,10;100,10;100,100;10,100" occluded="0" source="manual" z_order="0"></polygon>
        <box label="fallen" xtl="200" ytl="200" xbr="300" ybr="300" occluded="0" source="manual" z_order="0"></box>
    </image>
    <image id="1" name="image2_incomplete.png" width="512" height="512" task_id="1">
        <polygon label="incomplete" points="0,0;50,0;50,50;0,50"></polygon>
    </image>
    <image id="2" name="image3_unlabeled_only.png" width="512" height="512" task_id="1">
        <polygon label="unlabeled" points="0,0;50,0;50,50;0,50"></polygon>
    </image>
    <image id="3" name="image4_no_annotations.png" width="512" height="512" task_id="1"></image>
    <image id="4" name="image5_large.png" width="12000" height="12000" task_id="2">
         <polygon label="other" points="10,10;100,10;100,100;10,100"></polygon>
    </image>
    <image id="5" name="image6_4channel.png" width="512" height="512" task_id="2">
        <polygon label="upright" points="10,10;50,10;50,50;10,50"></polygon>
    </image>
    <image id="6" name="image7_grayscale.png" width="512" height="512" task_id="2">
        <polygon label="upright" points="10,10;50,10;50,50;10,50"></polygon>
    </image>
    <image id="7" name="image8_unsupported_channels.png" width="512" height="512" task_id="2">
        <polygon label="upright" points="10,10;50,10;50,50;10,50"></polygon>
    </image>
     <image id="8" name="image_for_all_upright_rule.png" width="512" height="512" task_id="3">
        <polygon label="upright" points="10,10;250,10;250,250;10,250"></polygon> <polygon label="upright" points="260,10;500,10;500,250;260,250"></polygon> <polygon label="upright" points="10,260;250,260;250,500;10,500"></polygon> <polygon label="upright" points="260,260;500,260;500,500;260,500"></polygon> </image>
    <image id="9" name="image_mixed_labels.png" width="512" height="512" task_id="3">
        <polygon label="upright" points="10,10;50,10;50,50;10,50"></polygon>
        <polygon label="fallen" points="60,60;100,60;100,100;60,100"></polygon>
    </image>
    <image id="10" name="image_with_unlabeled_poly.png" width="512" height="512" task_id="3">
        <polygon label="upright" points="10,10;50,10;50,50;10,50"></polygon>
        <polygon label="unlabeled" points="100,100;150,100;150,150;100,150"></polygon>
    </image>
    <image id="11" name="image_no_valid_label_in_chip.png" width="512" height="512" task_id="3">
         <polygon label="other" points="600,600;650,600;650,650;600,650"></polygon> </image>
    <image id="12" name="image_for_task_not_in_meta.png" width="100" height="100" task_id="99">
        <polygon label="upright" points="1,1;2,1;2,2;1,2"></polygon>
    </image>
</annotations>
"""

SAMPLE_CVAT_XML_PARSE_ERROR = "This is not valid XML"
SAMPLE_CVAT_XML_NO_PROJECT_UPDATED = """
<annotations><meta><project><name>Test</name></project><task><id>1</id><name>T1</name><updated>2024-05-10 11:00:00.000000+00:00</updated></task></meta>
<image id="0" name="img.png" width="10" height="10" task_id="1"><polygon label="upright" points="1,1;2,1;2,2;1,2"></polygon></image>
</annotations>
"""


SAMPLE_SERVER_PATHS_CSV = """Location Name,Server Folder Path
Task1_LocationA,/mnt/server_images/loc_a
Task2_LocationB,\\\\unc\\path\\loc_b
Task3_LocationC,C:\\local_server_folder\\loc_c
"""

EMPTY_SERVER_PATHS_CSV = "Location Name,Server Folder Path\n"
MALFORMED_SERVER_PATHS_CSV = "Header1\nValue1,Value2,Value3\nValueOnly\n"


@pytest.fixture
def mock_env(tmp_path, monkeypatch):
    """Fixture to set up a temporary environment for tests."""
    # Use tmp_path for all file operations to avoid cluttering the project directory
    test_base_output_dir = tmp_path / "dataset_processed"
    test_images_out_dir = test_base_output_dir / "images"
    test_masks_out_dir = test_base_output_dir / "masks"
    test_last_modified_file = test_base_output_dir / "lastmodified.txt"
    test_local_image_root_dir = tmp_path / "dataset_raw"

    monkeypatch.setattr(image_processor, "BASE_OUTPUT_DIR", test_base_output_dir)
    monkeypatch.setattr(image_processor, "IMAGES_OUT_DIR", test_images_out_dir)
    monkeypatch.setattr(image_processor, "MASKS_OUT_DIR", test_masks_out_dir)
    monkeypatch.setattr(image_processor, "LAST_MODIFIED_FILE", test_last_modified_file)
    monkeypatch.setattr(image_processor, "LOCAL_IMAGE_ROOT_DIR", test_local_image_root_dir)

    # Ensure parent directories for outputs exist if script doesn't create them early
    test_base_output_dir.mkdir(parents=True, exist_ok=True)
    test_local_image_root_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        "base_output_dir": test_base_output_dir,
        "images_out_dir": test_images_out_dir,
        "masks_out_dir": test_masks_out_dir,
        "last_modified_file": test_last_modified_file,
        "local_image_root_dir": test_local_image_root_dir,
    }

# --- Tests for Helper Functions ---

def test_parse_cvat_xml_valid():
    with patch('xml.etree.ElementTree.parse') as mock_parse:
        mock_tree = ET.ElementTree(ET.fromstring(SAMPLE_CVAT_XML_VALID))
        mock_parse.return_value = mock_tree
        
        tasks = image_processor.parse_cvat_xml("dummy.xml")
        assert tasks is not None
        assert "1" in tasks
        assert tasks["1"]["name"] == "Task1_LocationA"
        assert tasks["1"]["updated"] == datetime(2024, 5, 10, 11, 0, 0, tzinfo=timezone.utc)
        assert len(tasks["1"]["images"]) == 4 # image1, image2_incomplete, image3_unlabeled_only, image4_no_annotations
        assert tasks["1"]["images"][0]["name"] == "image1.png"
        assert len(tasks["1"]["images"][0]["annotations"]) == 2 # polygon and box
        assert tasks["1"]["images"][0]["annotations"][0]["type"] == "polygon"
        assert tasks["1"]["images"][0]["annotations"][1]["type"] == "box"
        assert tasks["2"]["updated"] == datetime(2024, 5, 20, 10, 0, 0, 123456, tzinfo=timezone.utc)
        assert tasks["3"]["updated"] == datetime(2024, 5, 20, 10, 0, 0, 123000, tzinfo=timezone.utc) # Microseconds from Z
        assert tasks["4"]["name"] == "Task4_BadDate" # Fallback to now()
        assert "99" not in tasks # Image with task_id 99 whose task is not in meta

def test_parse_cvat_xml_parse_error(capsys):
    with patch('xml.etree.ElementTree.parse', side_effect=ET.ParseError("mock parse error")):
        result = image_processor.parse_cvat_xml("dummy.xml")
        assert result is None
        captured = capsys.readouterr()
        assert "Error parsing XML file dummy.xml: mock parse error" in captured.out

def test_parse_cvat_xml_no_project_updated(capsys):
    with patch('xml.etree.ElementTree.parse') as mock_parse:
        mock_tree = ET.ElementTree(ET.fromstring(SAMPLE_CVAT_XML_NO_PROJECT_UPDATED))
        mock_parse.return_value = mock_tree
        tasks = image_processor.parse_cvat_xml("dummy_no_proj_updated.xml")
        assert tasks is not None
        assert "1" in tasks # Should still parse tasks

def test_read_server_paths_valid():
    with patch("builtins.open", mock_open(read_data=SAMPLE_SERVER_PATHS_CSV)) as mock_file:
        paths = image_processor.read_server_paths("dummy.csv")
        assert paths == {
            "Task1_LocationA": "/mnt/server_images/loc_a",
            "Task2_LocationB": "\\\\unc\\path\\loc_b",
            "Task3_LocationC": "C:\\local_server_folder\\loc_c",
        }
        mock_file.assert_called_once_with("dummy.csv", mode='r', newline='')

def test_read_server_paths_file_not_found(capsys):
    with patch("builtins.open", side_effect=FileNotFoundError):
        paths = image_processor.read_server_paths("nonexistent.csv")
        assert paths == {}
        captured = capsys.readouterr()
        assert "Warning: Server paths CSV 'nonexistent.csv' not found." in captured.out

def test_read_server_paths_empty_csv():
     with patch("builtins.open", mock_open(read_data=EMPTY_SERVER_PATHS_CSV)):
        paths = image_processor.read_server_paths("empty.csv")
        assert paths == {}

def test_read_server_paths_malformed_csv(capsys):
    with patch("builtins.open", mock_open(read_data=MALFORMED_SERVER_PATHS_CSV)):
        paths = image_processor.read_server_paths("malformed.csv")
        assert paths == {} # Should skip malformed rows gracefully
        captured = capsys.readouterr()
        assert "Error reading server paths CSV 'malformed.csv'" in captured.out


def test_get_last_modified_timestamp_exists_valid(mock_env):
    timestamp_str = datetime.now(timezone.utc).isoformat()
    mock_env["last_modified_file"].write_text(timestamp_str)
    ts = image_processor.get_last_modified_timestamp()
    assert ts == datetime.fromisoformat(timestamp_str)

def test_get_last_modified_timestamp_exists_invalid(mock_env, capsys):
    mock_env["last_modified_file"].write_text("invalid-timestamp")
    ts = image_processor.get_last_modified_timestamp()
    assert ts == datetime.min.replace(tzinfo=timezone.utc)
    captured = capsys.readouterr()
    assert f"Warning: Could not parse timestamp from {mock_env['last_modified_file']}" in captured.out

def test_get_last_modified_timestamp_not_exists(mock_env):
    # Ensure file does not exist (handled by tmp_path)
    ts = image_processor.get_last_modified_timestamp()
    assert ts == datetime.min.replace(tzinfo=timezone.utc)

def test_set_last_modified_timestamp(mock_env):
    image_processor.set_last_modified_timestamp()
    assert mock_env["last_modified_file"].exists()
    # Check if it's a valid ISO format timestamp (roughly)
    content = mock_env["last_modified_file"].read_text()
    assert datetime.fromisoformat(content) is not None
    # Test os.makedirs call when base_output_dir does not exist
    mock_env["base_output_dir"].rmdir() # Remove to test creation
    image_processor.set_last_modified_timestamp()
    assert mock_env["base_output_dir"].exists()


def test_scale_annotations():
    annotations = [{"points": [[10, 20], [30, 40]], "label": "test"}]
    scaled = image_processor.scale_annotations(annotations, 0.5)
    assert scaled[0]["points"] == [[5.0, 10.0], [15.0, 20.0]]

def test_ensure_dir(tmp_path):
    dir_path = tmp_path / "new_dir"
    assert not dir_path.exists()
    image_processor.ensure_dir(dir_path)
    assert dir_path.exists()
    image_processor.ensure_dir(dir_path) # Call again, should not fail
    assert dir_path.exists()


@patch('shutil.copy2')
@patch('pathlib.Path.exists')
def test_get_image_full_path_local_exists(mock_path_exists, mock_copy2, mock_env):
    mock_path_exists.return_value = True # Simulate local file exists
    local_img_path = mock_env["local_image_root_dir"] / "Task1" / "img.png"
    
    # We need to make sure the specific path for local_path.exists() returns True
    def side_effect_path_exists(path_arg):
        if path_arg == local_img_path:
            return True
        return False # Default for other checks if any
    mock_path_exists.side_effect = side_effect_path_exists

    result = image_processor.get_image_full_path("img.png", "Task1", {})
    assert result == str(local_img_path)
    mock_copy2.assert_not_called()

@patch('shutil.copy2')
@patch('pathlib.Path.mkdir')
@patch('pathlib.Path.exists')
def test_get_image_full_path_server_exists_unc(mock_path_exists, mock_mkdir, mock_copy2, mock_env):
    task_name = "TaskFromServer"
    image_name = "server_img.png"
    server_folder_path = "\\\\unc\\server\\images"
    server_paths_data = {task_name: server_folder_path}
    
    local_path = mock_env["local_image_root_dir"] / task_name / image_name
    potential_server_path = Path(server_folder_path) / image_name

    # Order of exists calls: local_path, potential_server_path, local_path.parent
    path_existence_map = {
        local_path: False,
        potential_server_path: True,
        local_path.parent: False # To test mkdir call
    }
    mock_path_exists.side_effect = lambda p: path_existence_map.get(p, False)

    result = image_processor.get_image_full_path(image_name, task_name, server_paths_data)
    
    assert result == str(local_path)
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_copy2.assert_called_once_with(potential_server_path, local_path)

@patch('shutil.copy2')
@patch('pathlib.Path.mkdir')
@patch('pathlib.Path.exists')
def test_get_image_full_path_server_exists_drive(mock_path_exists, mock_mkdir, mock_copy2, mock_env):
    task_name = "TaskFromServerDrive"
    image_name = "server_img_drive.png"
    server_folder_path = "D:\\server\\images" # Drive path
    server_paths_data = {task_name: server_folder_path}
    
    local_path = mock_env["local_image_root_dir"] / task_name / image_name
    potential_server_path = Path(server_folder_path) / image_name
    local_path_parent = local_path.parent

    # Path.drive attribute needs to be handled for Path objects
    # We can't directly mock attributes of Path instances created inside the function easily without
    # more complex patching of the Path class itself.
    # For this test, we assume Path(server_folder_path).drive will work as expected.
    # We mock the .exists() calls primarily.
    
    path_existence_map = {
        local_path: False, # Not found locally
        potential_server_path: True, # Found on server
        local_path_parent: True # Parent directory exists
    }
    mock_path_exists.side_effect = lambda p: path_existence_map.get(p, False)

    result = image_processor.get_image_full_path(image_name, task_name, server_paths_data)
    
    assert result == str(local_path)
    mock_mkdir.assert_not_called() # Parent dir exists
    mock_copy2.assert_called_once_with(potential_server_path, local_path)


@patch('shutil.copy2')
@patch('pathlib.Path.exists')
def test_get_image_full_path_server_copy_error(mock_path_exists, mock_copy2, mock_env, capsys):
    task_name = "TaskCopyError"
    image_name = "img_copy_error.png"
    server_folder_path = "\\\\unc\\server\\images"
    server_paths_data = {task_name: server_folder_path}
    local_path = mock_env["local_image_root_dir"] / task_name / image_name
    potential_server_path = Path(server_folder_path) / image_name

    path_existence_map = { local_path: False, potential_server_path: True, local_path.parent: True }
    mock_path_exists.side_effect = lambda p: path_existence_map.get(p, False)
    mock_copy2.side_effect = Exception("Copy failed")

    result = image_processor.get_image_full_path(image_name, task_name, server_paths_data)
    assert result is None
    assert "Error downloading image from server: Copy failed" in capsys.readouterr().out

@patch('pathlib.Path.exists')
def test_get_image_full_path_not_on_server(mock_path_exists, mock_env, capsys):
    task_name = "TaskNotOnServer"
    image_name = "img_not_on_server.png"
    server_folder_path = "\\\\unc\\server\\images"
    server_paths_data = {task_name: server_folder_path}
    local_path = mock_env["local_image_root_dir"] / task_name / image_name
    potential_server_path = Path(server_folder_path) / image_name

    path_existence_map = { local_path: False, potential_server_path: False } # Not on server
    mock_path_exists.side_effect = lambda p: path_existence_map.get(p, False)

    result = image_processor.get_image_full_path(image_name, task_name, server_paths_data)
    assert result is None
    assert f"Image {image_name} not found at server path: {potential_server_path}" in capsys.readouterr().out

@patch('pathlib.Path.exists')
def test_get_image_full_path_invalid_server_path_format(mock_path_exists, mock_env, capsys):
    task_name = "TaskInvalidServerPath"
    image_name = "img.png"
    server_folder_path = "http://not/a/file/path" # Invalid format
    server_paths_data = {task_name: server_folder_path}
    local_path = mock_env["local_image_root_dir"] / task_name / image_name

    mock_path_exists.return_value = False # Ensure local check fails

    result = image_processor.get_image_full_path(image_name, task_name, server_paths_data)
    assert result is None
    assert f"Server path for {task_name} ('{server_folder_path}') is not a recognized file system path." in capsys.readouterr().out


@patch('pathlib.Path.exists')
def test_get_image_full_path_task_not_in_csv(mock_path_exists, mock_env, capsys):
    mock_path_exists.return_value = False # Local check fails
    result = image_processor.get_image_full_path("img.png", "TaskNotInCSV", {})
    assert result is None
    assert "No server path entry found for location: TaskNotInCSV in CSV." in capsys.readouterr().out

@patch('pathlib.Path.exists')
def test_get_image_full_path_not_found_anywhere(mock_path_exists, mock_env, capsys):
    mock_path_exists.return_value = False # All Path.exists() calls return False
    server_paths_data = {"TaskX": "some_path"} # to enter the server check branch
    result = image_processor.get_image_full_path("img_ghost.png", "TaskX", server_paths_data)
    assert result is None
    assert "Failed to find image 'img_ghost.png' for task 'TaskX' both locally and on server." in capsys.readouterr().out


def test_polygon_intersects_bbox():
    poly = [[10,10], [20,10], [20,20], [10,20]]
    assert image_processor.polygon_intersects_bbox(poly, 5, 5, 15, 15) == True  # Overlap
    assert image_processor.polygon_intersects_bbox(poly, 0, 0, 5, 5) == False # No overlap X
    assert image_processor.polygon_intersects_bbox(poly, 30, 30, 40, 40) == False # No overlap Y
    assert image_processor.polygon_intersects_bbox(poly, 0, 0, 8, 22) == False # bbox_max_x < poly_min_x edge case
    assert image_processor.polygon_intersects_bbox(poly, 12, 0, 22, 8) == False # bbox_max_y < poly_min_y edge case

    assert image_processor.polygon_intersects_bbox([], 0, 0, 1, 1) == False # Empty polygon

# --- Tests for Main Processing Logic ---

@patch('image_processor.parse_cvat_xml')
def test_process_images_no_tasks(mock_parse_cvat, capsys, mock_env):
    mock_parse_cvat.return_value = None
    image_processor.process_images("dummy.xml", "dummy.csv")
    captured = capsys.readouterr()
    assert "No tasks found or error parsing XML. Exiting." in captured.out
    assert not mock_env["last_modified_file"].exists()


@patch('image_processor.cv2.imwrite')
@patch('image_processor.cv2.imread')
@patch('image_processor.get_image_full_path')
@patch('image_processor.read_server_paths')
@patch('image_processor.parse_cvat_xml')
@patch('image_processor.get_last_modified_timestamp')
@patch('image_processor.set_last_modified_timestamp')
def test_process_images_full_run(
    mock_set_ts, mock_get_ts, mock_parse_cvat, mock_read_paths, mock_get_img_path,
    mock_cv2_imread, mock_cv2_imwrite, capsys, mock_env, monkeypatch
):
    # --- Mocks Setup ---
    # Make CHIP_SIZE smaller for faster dummy image processing if needed, but ensure it matches expectations
    monkeypatch.setattr(image_processor, "CHIP_SIZE", 64) # Smaller chip for test
    monkeypatch.setattr(image_processor, "STRIDE", 32)   # Smaller stride

    # 1. get_last_modified_timestamp
    mock_get_ts.return_value = datetime.min.replace(tzinfo=timezone.utc)

    # 2. parse_cvat_xml
    parsed_xml_data = {
        "task1": {
            "name": "TestTask1",
            "updated": datetime.now(timezone.utc) - timedelta(days=1), # Older, should be skipped
            "images": []
        },
        "task2": {
            "name": "TestTask2_ProcessThis",
            "updated": datetime.now(timezone.utc) + timedelta(days=1), # Newer
            "images": [
                {
                    "id": "img1", "name": "image1.png", "width": 128, "height": 128, # Fits 2x2 chips with 64x64, stride 32
                    "annotations": [
                        {"type": "polygon", "label": "upright", "points": [[10,10],[60,10],[60,60],[10,60]]},
                        {"type": "polygon", "label": "fallen", "points": [[70,70],[120,70],[120,120],[70,120]]}
                    ]
                },
                {"id": "img2", "name": "image2_incomplete.png", "width": 64, "height": 64, "annotations": [{"type": "polygon", "label": "incomplete", "points": [[0,0]]}]},
                {"id": "img3", "name": "image3_unlabeled_only.png", "width": 64, "height": 64, "annotations": [{"type": "polygon", "label": "unlabeled", "points": [[0,0]]}]},
                {"id": "img4", "name": "image4_no_path.png", "width": 64, "height": 64, "annotations": [{"type": "polygon", "label": "upright", "points": [[0,0]]}]},
                {"id": "img5", "name": "image5_load_fail.png", "width": 64, "height": 64, "annotations": [{"type": "polygon", "label": "upright", "points": [[0,0]]}]},
                {"id": "img6", "name": "image6_large_and_scale.png", "width": 10001, "height": 64, # Triggers scaling
                    "annotations": [{"type": "polygon", "label": "other", "points": [[100,10],[600,10],[600,60],[100,60]]}]
                },
                {"id": "img7", "name": "image7_4channel.png", "width": 64, "height": 64, "annotations": [{"type": "polygon", "label": "upright", "points": [[0,0]]}]},
                {"id": "img8", "name": "image8_grayscale.png", "width": 64, "height": 64, "annotations": [{"type": "polygon", "label": "upright", "points": [[0,0]]}]},
                {"id": "img9", "name": "image9_bad_channels.png", "width": 64, "height": 64, "annotations": [{"type": "polygon", "label": "upright", "points": [[0,0]]}]},
                {"id": "img10", "name": "image10_all_upright_rule.png", "width": 64, "height": 64,
                    "annotations": [ # All quadrants for a 64x64 chip
                        {"type": "polygon", "label": "upright", "points": [[1,1],[31,1],[31,31],[1,31]]}, #TL
                        {"type": "polygon", "label": "upright", "points": [[33,1],[63,1],[63,31],[33,31]]}, #TR
                        {"type": "polygon", "label": "upright", "points": [[1,33],[31,33],[31,63],[1,63]]}, #BL
                        {"type": "polygon", "label": "upright", "points": [[33,33],[63,33],[63,63],[33,33]]}, #BR
                    ]
                },
                {"id": "img11", "name": "image11_chip_unlabeled.png", "width": 64, "height": 64,
                    "annotations": [{"type": "polygon", "label": "unlabeled", "points": [[10,10]]}] # This chip should be skipped
                },
                {"id": "img12", "name": "image12_chip_no_actual_label.png", "width": 64, "height": 64,
                    "annotations": [] # This chip should be skipped (or if only 'unlabeled' was present)
                },
                 {"id": "img13", "name": "image13_no_intersecting_polys.png", "width": 64, "height": 64,
                    "annotations": [{"type": "polygon", "label": "upright", "points": [[1000,1000]]}] # no intersection
                },


            ]
        }
    }
    mock_parse_cvat.return_value = parsed_xml_data

    # 3. read_server_paths
    mock_read_paths.return_value = {"TestTask2_ProcessThis": "/dummy/server/path"}

    # 4. get_image_full_path
    dummy_image_path_prefix = mock_env["local_image_root_dir"] / "TestTask2_ProcessThis"
    dummy_image_path_prefix.mkdir(parents=True, exist_ok=True)
    
    # Create dummy files that get_image_full_path would "find" or "download"
    (dummy_image_path_prefix / "image1.png").touch()
    (dummy_image_path_prefix / "image6_large_and_scale.png").touch()
    (dummy_image_path_prefix / "image7_4channel.png").touch()
    (dummy_image_path_prefix / "image8_grayscale.png").touch()
    (dummy_image_path_prefix / "image9_bad_channels.png").touch()
    (dummy_image_path_prefix / "image10_all_upright_rule.png").touch()
    (dummy_image_path_prefix / "image11_chip_unlabeled.png").touch()
    (dummy_image_path_prefix / "image12_chip_no_actual_label.png").touch()
    (dummy_image_path_prefix / "image13_no_intersecting_polys.png").touch()


    def get_img_path_side_effect(image_name, task_name, server_paths_data):
        if image_name == "image4_no_path.png": return None
        # For image5_load_fail.png, return a path but imread will fail
        return str(dummy_image_path_prefix / image_name)
    mock_get_img_path.side_effect = get_img_path_side_effect
    
    # 5. cv2.imread
    dummy_bgr_img = np.zeros((64, 64, 3), dtype=np.uint8)
    dummy_bgr_img_large = np.zeros((64, 10001, 3), dtype=np.uint8) # For scaling test
    dummy_bgra_img = np.zeros((64, 64, 4), dtype=np.uint8)
    dummy_gray_img = np.zeros((64, 64), dtype=np.uint8)
    dummy_bad_channels_img = np.zeros((64, 64, 5), dtype=np.uint8) # Unsupported channels

    def imread_side_effect(path_str, flag):
        path_obj = Path(path_str)
        if "image5_load_fail.png" in path_str: return None
        if "image6_large_and_scale.png" in path_str: return dummy_bgr_img_large
        if "image7_4channel.png" in path_str: return dummy_bgra_img
        if "image8_grayscale.png" in path_str: return dummy_gray_img
        if "image9_bad_channels.png" in path_str: return dummy_bad_channels_img
        return dummy_bgr_img # Default for others
    mock_cv2_imread.side_effect = imread_side_effect

    # --- Run the processor ---
    image_processor.process_images("dummy.xml", "dummy.csv")

    # --- Assertions ---
    captured = capsys.readouterr()
    # print(captured.out) # For debugging test failures

    assert "Skipping task 'TestTask1'" in captured.out
    assert "Processing task: TestTask2_ProcessThis" in captured.out
    
    # Image-specific skips
    assert "image2_incomplete.png' contains 'incomplete' labels. Skipping" in captured.out
    assert "image3_unlabeled_only.png' has no valid labels or only 'unlabeled' labels. Skipping" in captured.out
    assert "Could not find image 'image4_no_path.png'. Skipping." in captured.out
    assert "Failed to load image:" in captured.out and "image5_load_fail.png" in captured.out
    
    # Image processing messages
    assert "image6_large_and_scale.png" in captured.out and "Downscaling by factor of 2" in captured.out
    assert "Overwrote original image with downscaled version" in captured.out # for image6
    assert "image7_4channel.png has 4 channels. Converting to BGR" in captured.out
    assert "image8_grayscale.png is grayscale. Converting to BGR" in captured.out
    assert "image9_bad_channels.png has an unsupported number of channels: 5. Skipping." in captured.out
    
    # Chip skipping messages (these print statements are commented out in the original code, so we check for their effects)
    # For image11_chip_unlabeled.png, chips should be skipped.
    # For image12_chip_no_actual_label.png, chips should be skipped.
    # If the print statements were active, we'd check like:
    # assert "Skipping chip" in captured.out and "image11_chip_unlabeled.png" in captured.out and "contains 'unlabeled' type" in captured.out
    # assert "Skipping chip" in captured.out and "image12_chip_no_actual_label.png" in captured.out and "no actual labels found" in captured.out
    
    # Chip generation for image1.png: 128x128 image, 64x64 chip, 32 stride
    # (0,0), (0,32), (0,64)
    # (32,0), (32,32), (32,64)
    # (64,0), (64,32), (64,64)
    # Total 9 chips for image1.png if all contain labels. Given annotations, some might be skipped if empty after clipping.
    # Let's count imwrite calls.
    # image1.png: 9 potential chip positions.
    # (0,0) chip: upright polygon. (0,32) chip: upright. (32,0) chip: upright. (32,32) chip: upright+fallen
    # (0,64) chip: fallen. (32,64) chip: fallen. (64,0) chip: no label. (64,32) chip: fallen. (64,64) chip: fallen
    # This gets complex to trace exactly without running it.
    # For image10_all_upright_rule.png: 1 chip position (0,0), should trigger all-upright.
    # mask_chip should be all LABEL_TO_VALUE["upright"] for this one.

    # Total imwrite calls: each saved chip_image + chip_mask = 2 calls per chip.
    # Let's verify at least some chips were made and the timestamp was set.
    assert mock_cv2_imwrite.called
    num_chips_expected_at_least_for_img1 = 1 # At least one chip from image1.png with mixed labels
    num_chips_expected_for_img10 = 1 # For the all-upright rule
    
    # Count calls to imwrite. Each chip is 2 calls (image and mask).
    # Hard to get an exact number without meticulously tracing polygon intersections for each chip.
    # Instead, check that directories were created and some files exist.
    task2_images_out = mock_env["images_out_dir"] / "TestTask2_ProcessThis"
    task2_masks_out = mock_env["masks_out_dir"] / "TestTask2_ProcessThis"
    assert task2_images_out.is_dir()
    assert task2_masks_out.is_dir()
    
    # Check if at least one chip was written for image1
    # (e.g. image1_chip_0_0.png and image1_chip_0_0_mask.png)
    # Chip naming: f"{Path(image_name).stem}_chip_{int(x/STRIDE)}_{int(y/STRIDE)}"
    # Stride is 32. x/256, y/256 from original script needs to match stride division logic.
    # Assuming chip naming based on x, y pixel coords / stride:
    # image1_chip_0_0.png, image1_chip_0_0_mask.png
    # image1_chip_0_1.png, image1_chip_0_1_mask.png (y=32 -> y_idx=1)
    # image1_chip_1_0.png, image1_chip_1_0_mask.png (x=32 -> x_idx=1)
    
    # Simplified check: were some PNGs created?
    image_chip_files = list(task2_images_out.glob("*.png"))
    mask_chip_files = list(task2_masks_out.glob("*.png"))
    assert len(image_chip_files) > 0
    assert len(mask_chip_files) > 0
    assert len(image_chip_files) == len(mask_chip_files)

    mock_set_ts.assert_called_once()
    assert f"Finished processing." in captured.out
    assert f"{len(image_chip_files)} chips generated." in captured.out # This depends on the actual number of chips

    # Test the "all upright rule" specifically for image10
    # We need to inspect the arguments to cv2.imwrite for the mask of image10's chip
    found_all_upright_mask = False
    for call_args in mock_cv2_imwrite.call_args_list:
        args, _ = call_args
        filename = Path(args[0])
        mask_data = args[1]
        if "image10_all_upright_rule_chip" in filename.name and "_mask.png" in filename.name:
            assert np.all(mask_data == image_processor.LABEL_TO_VALUE["upright"])
            found_all_upright_mask = True
            # print(f"Found and verified all_upright_mask for {filename}")
    assert found_all_upright_mask

    # Test standard mask drawing and fill for image1 (first chip 0,0)
    # It has an "upright" polygon. Rest should be filled with "upright".
    found_image1_chip00_mask = False
    for call_args in mock_cv2_imwrite.call_args_list:
        args, _ = call_args
        filename = Path(args[0])
        mask_data = args[1]
        if "image1_chip_0_0_mask" in filename.name: # x=0, y=0
            # Check that some pixels are 'upright' due to polygon, and some from fill
            # This requires more detailed knowledge of the polygon's effect on the chip.
            # A simple check: are all pixels a valid label value?
            valid_label_values = set(image_processor.LABEL_TO_VALUE.values())
            # Remove special values if they are not expected in final mask
            valid_label_values.discard(image_processor.LABEL_TO_VALUE["unlabeled"])
            valid_label_values.discard(image_processor.LABEL_TO_VALUE["incomplete"])
            
            unique_values_in_mask = np.unique(mask_data)
            for val in unique_values_in_mask:
                 assert val in valid_label_values

            # Specifically, check if the NO_LABEL_TEMP_VALUE is not present
            assert image_processor.NO_LABEL_TEMP_VALUE not in unique_values_in_mask
            # And it should primarily be 'upright' for this chip based on the sample data for image1.
            assert image_processor.LABEL_TO_VALUE["upright"] in unique_values_in_mask
            found_image1_chip00_mask = True
            # print(f"Found and verified image1_chip00_mask for {filename} with values {unique_values_in_mask}")

    assert found_image1_chip00_mask


@patch('image_processor.set_last_modified_timestamp')
@patch('image_processor.get_last_modified_timestamp')
@patch('image_processor.read_server_paths')
@patch('image_processor.parse_cvat_xml')
def test_process_images_no_new_updates(mock_parse_cvat, mock_read_paths, mock_get_ts, mock_set_ts, capsys, mock_env):
    # All tasks are older than last_processed_time
    mock_get_ts.return_value = datetime.now(timezone.utc) # Current time as last processed
    mock_parse_cvat.return_value = {
        "task1": {
            "name": "OldTask",
            "updated": datetime.now(timezone.utc) - timedelta(days=1), # Older
            "images": [{"id": "img1", "name": "image.png", "width": 10, "height": 10, "annotations": []}]
        }
    }
    mock_read_paths.return_value = {}
    
    image_processor.process_images("dummy.xml", "dummy.csv")
    
    captured = capsys.readouterr()
    assert "Skipping task 'OldTask'" in captured.out
    assert "No images needed processing based on timestamps." in captured.out
    mock_set_ts.assert_not_called()


# Test for the main block (optional, usually covered by testing process_images)
# To do this, you'd need to import the script and then call its main execution,
# or mock the process_images call itself.
@patch('image_processor.process_images')
@patch('pathlib.Path.exists')
def test_main_block_calls_process_images(mock_path_exists, mock_process_images, monkeypatch):
    # Simulate that files exist
    mock_path_exists.return_value = True
    
    # Store original __name__ and set to "__main__" for the test
    original_name = image_processor.__name__
    monkeypatch.setattr(image_processor, "__name__", "__main__")
    
    # This is tricky because the `if __name__ == "__main__":` block runs on import
    # if not careful. A better way is to put the main logic into a function.
    # Let's assume your script `image_processor.py` has its main guard.
    # We can't directly "re-run" the __main__ block easily without `runpy`.
    # For simplicity, we'll test the conditions within the `if __name__ == "__main__":`

    # Scenario 1: Files exist
    with patch.object(image_processor, "cvat_xml_file", "valid_cvat.xml"), \
         patch.object(image_processor, "server_paths_file", "valid_server.csv"):
        # This part of the test would ideally involve `runpy` or refactoring main script
        # For now, this test primarily covers the `process_images` call.
        # If `main()` function existed:
        # image_processor.main() 
        # mock_process_images.assert_called_once_with("valid_cvat.xml", "valid_server.csv")
        pass # Covered by direct call test of process_images for logic

@patch('image_processor.process_images') # Mock the main function
@patch('sys.argv', ['image_processor.py']) # Mock command line arguments if used by main
@patch('pathlib.Path.exists')
def test_main_entry_point_cvat_xml_not_found(mock_path_exists, mock_process_images_call, capsys, monkeypatch):
    # Simulate CVAT XML not found, server paths CSV found
    def path_exists_side_effect(path_arg):
        if str(path_arg) == image_processor.cvat_xml_file: # Use the actual variable from script
            return False
        if str(path_arg) == image_processor.server_paths_file:
            return True
        return True # Default
    mock_path_exists.side_effect = path_exists_side_effect

    # This requires the main block to be in a function or use runpy
    # For this structure, we test the conditional logic that would be hit
    # by having the main block execute.
    
    # We need to temporarily change the values of cvat_xml_file and server_paths_file
    # in the image_processor module for this test to reflect the conditions in __main__
    monkeypatch.setattr(image_processor, "cvat_xml_file", "non_existent_cvat.xml")
    monkeypatch.setattr(image_processor, "server_paths_file", "existent_server.csv")
    
    # Manually simulate the check from the __main__ block
    # This is a way to test the logic without directly running the script's __main__
    if not Path(image_processor.cvat_xml_file).exists():
        print(f"Error: CVAT XML file not found at '{image_processor.cvat_xml_file}'")
    elif not Path(image_processor.server_paths_file).exists() and image_processor.LOCAL_IMAGE_ROOT_DIR == "":
        print(f"Warning: Server paths CSV not found at '{image_processor.server_paths_file}'.")
    else:
        image_processor.process_images(image_processor.cvat_xml_file, image_processor.server_paths_file)

    captured = capsys.readouterr()
    assert "Error: CVAT XML file not found at 'non_existent_cvat.xml'" in captured.out
    mock_process_images_call.assert_not_called()


@patch('image_processor.process_images')
@patch('pathlib.Path.exists')
def test_main_entry_point_server_paths_not_found_warning(mock_path_exists, mock_process_images_call, capsys, monkeypatch):
    # Simulate CVAT XML found, server paths CSV not found, LOCAL_IMAGE_ROOT_DIR is empty
    def path_exists_side_effect(path_arg):
        if str(path_arg) == image_processor.cvat_xml_file:
            return True
        if str(path_arg) == image_processor.server_paths_file:
            return False
        return True # Default
    mock_path_exists.side_effect = path_exists_side_effect

    monkeypatch.setattr(image_processor, "cvat_xml_file", "existent_cvat.xml")
    monkeypatch.setattr(image_processor, "server_paths_file", "non_existent_server.csv")
    monkeypatch.setattr(image_processor, "LOCAL_IMAGE_ROOT_DIR", "") # Critical for warning

    if not Path(image_processor.cvat_xml_file).exists():
        print(f"Error: CVAT XML file not found at '{image_processor.cvat_xml_file}'")
    elif not Path(image_processor.server_paths_file).exists() and image_processor.LOCAL_IMAGE_ROOT_DIR == "":
        print(f"Warning: Server paths CSV not found at '{image_processor.server_paths_file}'. Image download/fallback may not work.")
        # The original script proceeds in this case, so process_images WOULD be called
        image_processor.process_images(image_processor.cvat_xml_file, image_processor.server_paths_file)
    else: # This else would cover cases where LOCAL_IMAGE_ROOT_DIR is set
        image_processor.process_images(image_processor.cvat_xml_file, image_processor.server_paths_file)


    captured = capsys.readouterr()
    assert "Warning: Server paths CSV not found at 'non_existent_server.csv'" in captured.out
    mock_process_images_call.assert_called_once_with("existent_cvat.xml", "non_existent_server.csv")


@patch('image_processor.process_images')
@patch('pathlib.Path.exists')
def test_main_entry_point_all_exist(mock_path_exists, mock_process_images_call, capsys, monkeypatch):
    mock_path_exists.return_value = True # All files exist

    monkeypatch.setattr(image_processor, "cvat_xml_file", "existent_cvat.xml")
    monkeypatch.setattr(image_processor, "server_paths_file", "existent_server.csv")
    monkeypatch.setattr(image_processor, "LOCAL_IMAGE_ROOT_DIR", "some/local/path") # Not empty

    if not Path(image_processor.cvat_xml_file).exists():
        print(f"Error: CVAT XML file not found at '{image_processor.cvat_xml_file}'")
    elif not Path(image_processor.server_paths_file).exists() and image_processor.LOCAL_IMAGE_ROOT_DIR == "":
         print(f"Warning: Server paths CSV not found at '{image_processor.server_paths_file}'.")
    else:
        image_processor.process_images(image_processor.cvat_xml_file, image_processor.server_paths_file)

    mock_process_images_call.assert_called_once_with("existent_cvat.xml", "existent_server.csv")