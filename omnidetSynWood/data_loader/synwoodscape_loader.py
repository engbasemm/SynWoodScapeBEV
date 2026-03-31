"""
WoodScape Raw dataset loader class for OmniDet.

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import json
import os
import pickle
import random
from collections import namedtuple

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image  # using pillow-simd for increased speed
from torchvision import transforms

from train_utils.detection_utils import crop_annotation


class SynWoodScapeRawDataset(data.Dataset):
    """Fisheye Woodscape Raw dataloader"""

    def __init__(self, data_path=None, path_file=None, is_train=False, config=None, grid_config=None):
        super().__init__()

        self.data_path = data_path
        image_paths_full = [line.rstrip('\n') for line in open(path_file)]
        self.is_train = is_train
        self.args = config
        self.task = config.train
        self.batch_size = config.batch_size
        self.crop = config.crop
        self.semantic_classes = config.semantic_num_classes
        self.num_scales = config.num_scales
        self.frame_idxs = config.frame_idxs
        self.original_res = namedtuple('original_res', 'width height')(1280, 966)
        self.network_input_width = config.input_width
        self.network_input_height = config.input_height

        if grid_config:
            # Calculate BEV grid size from the grid_config if provided
            self.bev_grid_height = int((grid_config['ybound'][1] - grid_config['ybound'][0]) / grid_config['ybound'][2])
            self.bev_grid_width = int((grid_config['xbound'][1] - grid_config['xbound'][0]) / grid_config['xbound'][2])
        else:
            self.bev_grid_height = 420  # Fallback to a default value
            self.bev_grid_width = 300
        self.total_car1_images = 6054
        self.color_aug = None

        # Define the 4 SVS camera sides
        self.all_cam_sides = ['FV', 'MVL', 'MVR', 'RV']

        # Extract unique frame indices from the image_paths
        unique_frames = set()
        for img_path in image_paths_full:
            frame_index = img_path.split('.')[0].split('_')[0]
            unique_frames.add(frame_index)
        self.unique_frame_indices = sorted(list(unique_frames)) # Store sorted unique frame indices

        self.cropped_coords = dict(Car1=dict(FV=(114, 110, 1176, 610),
                                             MVL=(343, 5, 1088, 411),
                                             MVR=(185, 5, 915, 425),
                                             RV=(186, 203, 1105, 630)),
                                   Car2=dict(FV=(160, 272, 1030, 677),
                                             MVL=(327, 7, 1096, 410),
                                             MVR=(175, 4, 935, 404),
                                             RV=(285, 187, 1000, 572)))

        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((self.network_input_height, self.network_input_width),
                                        interpolation=transforms.InterpolationMode.BICUBIC)
        # Add a specific resize transform for the BEV ground truth, which was missing.
        self.resize_bev = transforms.Resize((self.bev_grid_height, self.bev_grid_width),
                                             interpolation=transforms.InterpolationMode.BICUBIC)
        self.resize_label = transforms.Resize((self.network_input_height, self.network_input_width),
                                              interpolation=transforms.InterpolationMode.NEAREST)
        # Corrected logical OR condition
        if "distance" in self.task or "depth" in self.task:
            # Corrected path for data files relative to the project root
            lut_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'LUTs.pkl')
            with open(lut_path, 'rb') as f:
                self.LUTs = pickle.load(f)

    def _parse_vehicle_txt(self, file_path):
        """Parses the custom vehicle data .txt file to extract ego_to_world matrix."""
        data = {}
        with open(file_path, 'r') as f:
            for line in f:
                if 'Transform' in line:
                    # Example line: Transform(Location(x=-111.0, y=80.9, z=-0.008), Rotation(pitch=0.1, yaw=88.2, roll=0.05))
                    try:
                        # This is a simplified parser. It assumes a fixed format.
                        # A more robust solution would use regular expressions.
                        parts = line.replace('Transform(Location(', '').replace('), Rotation(', ';').replace('))', '').split(';')
                        loc_parts = parts[0].replace('x=', '').replace(' y=', '').replace(' z=', '').split(',')
                        rot_parts = parts[1].replace('pitch=', '').replace(' yaw=', '').replace(' roll=', '').split(',')

                        x, y, z = [float(p) for p in loc_parts]
                        pitch, yaw, roll = [float(p) for p in rot_parts]

                        # Convert rotation (yaw, pitch, roll) to a rotation matrix
                        # This is a simplified conversion and might need adjustment based on the exact rotation order (e.g., ZYX)
                        cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
                        cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
                        cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))

                        # Assuming ZYX rotation order
                        rot_matrix = np.array([
                            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                            [-sp, cp*sr, cp*cr]
                        ])

                        # Create a 4x4 transformation matrix
                        transform_matrix = np.eye(4)
                        transform_matrix[:3, :3] = rot_matrix
                        transform_matrix[:3, 3] = [x, y, z]
                        data['ego_to_world'] = transform_matrix.tolist()
                    except (ValueError, IndexError) as e:
                        raise ValueError(f"Could not parse Transform line in {file_path}: {line}") from e
        if 'ego_to_world' not in data:
            raise ValueError(f"Could not find 'Transform' line to extract ego_to_world in {file_path}")
        return data

    def scale_intrinsic(self, intrinsics, cropped_coords) -> tuple:
        """Scales the intrinsics from original res to the network's initial input res"""
        D = np.array(intrinsics[4:8], dtype=np.float32)
        K = np.array(intrinsics[:3], dtype=np.float32)
        K = np.insert(K, 0, 1.0)
        K[2] += self.original_res.width / 2
        K[3] += self.original_res.height / 2
        if self.crop:
            # Adjust the offset of the cropped intrinsic around the width and height.
            K[2] -= cropped_coords[0]
            K[3] -= cropped_coords[1]
            # Compensate for resizing
            K[2] *= self.network_input_width / (cropped_coords[2] - cropped_coords[0])
            K[3] *= self.network_input_height / (cropped_coords[3] - cropped_coords[1])
            D *= self.network_input_width / (cropped_coords[2] - cropped_coords[0])
        else:
            D *= self.network_input_width / self.original_res.width
            K[2] *= self.network_input_width / self.original_res.width
            K[3] *= self.network_input_height / self.original_res.height
        return K, D

    def get_displacements_from_speed(self, frame_index, cam_side):
        """get displacement magnitudes using speed and time."""

        # This function seems unused in the BEV pipeline, but correcting it for completeness.
        # It would also need a custom parser if its files are not JSON.
        # For now, assuming it's not called and focusing on the get_extrinsics fix.
        try:
            previous_oxt_file = json.load(open(os.path.join(self.data_path, "vehicle_data", "previous_images", f'{frame_index}.json')))
            present_oxt_file = json.load(open(os.path.join(self.data_path, "vehicle_data", "rgb_images", f'{frame_index}.json')))
        except FileNotFoundError:
             # Fallback for txt if json is not found, assuming it might be needed by other tasks.
            previous_oxt_file = self._parse_vehicle_txt(os.path.join(self.data_path, "vehicle_data", "previous_images", f'{frame_index}.txt'))
            present_oxt_file = self._parse_vehicle_txt(os.path.join(self.data_path, "vehicle_data", "rgb_images", f'{frame_index}.txt'))

        timestamps = [float(previous_oxt_file["timestamp"]) / 1e6, float(present_oxt_file["timestamp"]) / 1e6]
        # Convert km/hr to m/s
        speeds_ms = [float(previous_oxt_file["ego_speed"]) / 3.6, float(present_oxt_file["ego_speed"]) / 3.6]

        displacement = np.array(0.5 * (speeds_ms[1] + speeds_ms[0]) * (timestamps[1] - timestamps[0])).astype(
            np.float32)

        return displacement

    def get_image(self, index, cropped_coords, frame_index, cam_side):
        recording_folder = self.args.rgb_images if index == 0 else "previous_images"
        file = f"{frame_index}_{cam_side}.png" if index == 0 else f"{frame_index}_{cam_side}_prev.png"
        path = os.path.join(self.data_path, recording_folder, file)
        image = Image.open(path).convert('RGB')
        if self.crop:
            return image.crop(cropped_coords)
        return image

    def get_bev_image(self, frame_index):
        """Helper function to load the BEV image."""
        path = os.path.join(self.data_path, self.args.rgb_images, f"{frame_index}_BEV.png")
        if os.path.exists(path):
            return Image.open(path).convert('RGB')
        else:
            # Return a black image if BEV is not found, to avoid crashing.
            print(f"Warning: BEV image not found at {path}. Returning a black image.")
            return Image.new('RGB', (self.original_res.width, self.original_res.height))

        # In data_loader/synwoodscape_loader.py, inside the SynWoodScapeRawDataset class:

    def get_extrinsics(self, frame_index, cam_side, frame_offset):
        """
        Calculates the extrinsic matrix from the camera to the ego-vehicle frame.
        This corrected version explicitly handles the coordinate system transformations
        from Unreal Engine (dataset) to the LSS-compatible frames (OpenCV camera, LSS ego).
        """
        # --- Load calibration data directly ---
        # Path for camera-specific calibration (cam_to_world)
        cam_calib_path = os.path.join(self.data_path, "calibration_data", f"{cam_side}.json")

        # Path for frame-specific vehicle data (ego_to_world)
        if frame_offset == 0:
            vehicle_data_folder = "rgb_images"
        else:  # Assuming frame_offset is -1 for previous frame
            vehicle_data_folder = "previous_images"

        # The vehicle data filename is a .txt file and is per-frame, not per-camera
        vehicle_data_path = os.path.join(self.data_path, "vehicle_data", vehicle_data_folder, f'{frame_index}.txt')

        if not os.path.exists(cam_calib_path):
            raise FileNotFoundError(f"Camera calibration file not found: {cam_calib_path}")
        if not os.path.exists(vehicle_data_path):
            raise FileNotFoundError(f"Vehicle data file not found: {vehicle_data_path}")

        with open(cam_calib_path) as f:
            cam_data = json.load(f)

        # Use the new custom parser for the vehicle data .txt file
        vehicle_data = self._parse_vehicle_txt(vehicle_data_path)

        # --- Build Camera-to-World Matrix from JSON data (Quaternion + Translation) ---
        extrinsic_data = cam_data['extrinsic']
        q = extrinsic_data['quaternion'] # Assuming [w, x, y, z] format
        t = extrinsic_data['translation used in CARLA (CARLA reference)']
        w, x, y, z = q[0], q[1], q[2], q[3]
        rot_matrix = torch.tensor([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ], dtype=torch.float32)
        cam_to_world_unreal = torch.eye(4, dtype=torch.float32)
        cam_to_world_unreal[:3, :3] = rot_matrix
        cam_to_world_unreal[:3, 3] = torch.tensor(t, dtype=torch.float32)

        # --- Build Ego-to-World Matrix from Parsed TXT data (Euler + Translation) ---
        ego_to_world_unreal = torch.tensor(vehicle_data['ego_to_world'], dtype=torch.float32)
        world_to_ego_unreal = torch.inverse(ego_to_world_unreal)

        # --- Define Coordinate System Transformations ---
        # T_lss_from_unreal_ego: Unreal Ego (X-Fwd, Y-Right, Z-Up) -> LSS Ego (X-Right, Y-Fwd, Z-Up).
        # The flip on the Y-axis mapping is the final correction.
        T_lss_from_unreal_ego = torch.tensor([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32)

        # T_unreal_cam_from_opencv: OpenCV Cam (X-Right, Y-Down, Z-Fwd) -> Unreal Cam (X-Fwd, Y-Right, Z-Up).
        T_unreal_cam_from_opencv = torch.tensor([[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32)

        extrinsics = (
            T_lss_from_unreal_ego @
            world_to_ego_unreal @
            cam_to_world_unreal @
            T_unreal_cam_from_opencv
        )
        return extrinsics


    def get_intrinsic_matrix(self, cropped_coords, frame_index, cam_side):
        """Loads camera intrinsics and returns them as a scaled 3x3 matrix."""
        # Reverted path to user-specified correct location.
        path = os.path.join(self.data_path, "calibration_data", f"{cam_side}.json")
        with open(path) as f:
            intr_data = json.load(f)['intrinsic']

        orig_w, orig_h = self.original_res.width, self.original_res.height
        net_w, net_h = self.network_input_width, self.network_input_height

        # Original intrinsics relative to top-left of original image
        fx_orig = intr_data['k1']
        fy_orig = intr_data['k1'] * intr_data['aspect_ratio']
        cx_orig = intr_data['cx_offset'] + orig_w / 2
        cy_orig = intr_data['cy_offset'] + orig_h / 2

        if self.crop and cropped_coords:
            crop_x1, crop_y1, crop_x2, crop_y2 = cropped_coords
            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1

            # Adjust principal point for the crop
            cx_cropped = cx_orig - crop_x1
            cy_cropped = cy_orig - crop_y1

            # Scale all parameters for the final resize from cropped to network size
            scale_w = net_w / crop_w
            scale_h = net_h / crop_h

            fx = fx_orig * scale_w
            fy = fy_orig * scale_h
            cx = cx_cropped * scale_w
            cy = cy_cropped * scale_h
        else:
            # Scale all parameters for the final resize from original to network size
            scale_w = net_w / orig_w
            scale_h = net_h / orig_h
            fx = fx_orig * scale_w
            fy = fy_orig * scale_h
            cx = cx_orig * scale_w
            cy = cy_orig * scale_h

        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    def get_label(self, gt_folder, cropped_coords, frame_index, cam_side):
        path = os.path.join(self.data_path, gt_folder, "gtLabels", f"{frame_index}_{cam_side}.png")
        image = Image.open(path).convert('L')
        if self.crop:
            return image.crop(cropped_coords)
        return image

    def get_intrinsics(self, cropped_coords, frame_index, cam_side):
        # Reverted path to user-specified correct location.
        data = json.load(open(os.path.join(self.data_path, "calibration_data", f"{cam_side}.json")))
        intrinsics = list(data['intrinsic'].values())
        K, D = self.scale_intrinsic(intrinsics, cropped_coords)
        return K, D, intrinsics

    def get_detection_label(self, crop_coords, frame_index, cam_side):
        path = os.path.join(self.data_path, "box_2d_annotations", f"{frame_index}_{cam_side}.txt")
        if os.path.exists(path) and os.stat(path).st_size != 0:  # Check if file exists and is not empty
            boxes = torch.from_numpy(np.loadtxt(path, delimiter=",", usecols=(1, 2, 3, 4, 5)).reshape(-1, 5))

            if boxes.numel() == 0:
                boxes = torch.from_numpy(np.zeros((1, 5)))
            else:
                # --- REMAPPING CLASS IDs TO 0 AND 1 FOR YOLO MODEL ---
                # Remap class ID 10 (car) to 0
                boxes[boxes[:, 0] == 10, 0] = 0
                # Remap class ID 4 (person) to 1
                boxes[boxes[:, 0] == 4, 0] = 1

                # Filter out boxes that are not now class 0 (car) or class 1 (person)
                newboxes = boxes[torch.where((boxes[:, 0] == 0) | (boxes[:, 0] == 1))[0]]

                if (len(newboxes) == 0):
                    boxes = torch.from_numpy(np.zeros((1, 5)))
                else:
                    carboxes = newboxes[newboxes[:, 0] == 0]
                    personboxes = newboxes[newboxes[:, 0] == 1]

                    # Apply minimum area filters
                    carboxes = carboxes[torch.abs(carboxes[:, 3] - carboxes[:, 1]) * torch.abs(carboxes[:, 4] - carboxes[:, 2]) > self.args.min_vehicle_area]
                    personboxes = personboxes[torch.abs(personboxes[:, 3] - personboxes[:, 1]) * torch.abs(personboxes[:, 4] - personboxes[:, 2]) > self.args.min_person_area]

                    # Collect and concatenate valid boxes
                    valid_boxes = []
                    if len(carboxes) > 0:
                        valid_boxes.append(carboxes)
                    if len(personboxes) > 0:
                        valid_boxes.append(personboxes)

                    if valid_boxes:
                        boxes = torch.cat(valid_boxes)
                    else:
                        boxes = torch.from_numpy(np.zeros((1, 5)))  # All filtered out
            # Re-parameterize box for annotation
            w = torch.abs(boxes[:, 3] - boxes[:, 1])
            h = torch.abs(boxes[:, 4] - boxes[:, 2])
            x_c = torch.minimum(boxes[:, 1], boxes[:, 3]) + (w / 2)
            y_c = torch.minimum(boxes[:, 2], boxes[:, 4]) + (h / 2)

            # # Normalize
            w /= self.original_res.width
            x_c /= self.original_res.width
            h /= self.original_res.height
            y_c /= self.original_res.height

            boxes[:, 1] = x_c
            boxes[:, 2] = y_c
            boxes[:, 3] = w
            boxes[:, 4] = h

            if self.crop and crop_coords:
                cropping = dict(left=crop_coords[0], top=crop_coords[1], right=crop_coords[2], bottom=crop_coords[3])
                cropped_boxes = crop_annotation(boxes, cropping, accepted_crop_ratio=0.4,
                                                orginial_image_size=self.original_res,
                                                img_size=(self.network_input_width, self.network_input_height),
                                                enable_scaling=True)
                boxes = cropped_boxes
        else:
            boxes = torch.from_numpy(np.zeros((1, 5)))

        targets = torch.zeros((len(boxes), 7))
        targets[:, 2:] = boxes
        return targets

    def to_tensor_semantic_label(self, label: np.array) -> torch.LongTensor:
        label[label > self.semantic_classes - 1] = 0
        return torch.LongTensor(label)

    @staticmethod
    def to_tensor_motion_label(label: np.array) -> torch.LongTensor:
        label[label > 0] = 1  # Any class greater than 0 is set to 1
        return torch.LongTensor(label)

    def preprocess(self, inputs):
        """Resize color images to the required scales and augment if required.
        Create the color_aug object in advance and apply the same augmentation to all images in this item.
        This ensures that all images input to the pose network receive the same augmentation.
        """
        labels_list = ["motion_labels", "semantic_labels"]

        # First pass: Handle resizing from native resolution (-1) to scale 0
        # Iterate over a copy of keys to avoid issues when adding new keys
        keys_to_process_for_resize = [k for k in list(inputs.keys()) if (len(k) > 2 and k[2] == -1)]
        for k in keys_to_process_for_resize:
            if "color" in k[0]:
                # Key is like ("color", frame_id, -1, cam_side)
                name, frame_id, _, cam_side = k
                inputs[(name, frame_id, 0, cam_side)] = self.resize(inputs[k])
            elif "bev_gt" in k[0]:
                # Key is like ("bev_gt", 0, -1)
                name, frame_id, _ = k
                # Use the new resize transform for BEV
                inputs[(name, frame_id, 0)] = self.resize_bev(inputs[k])
            elif any(x in k[0] for x in labels_list):
                # Key is like ("semantic_labels", frame_id, -1, cam_side)
                name, frame_id, _, cam_side = k
                inputs[(name, frame_id, 0, cam_side)] = self.resize_label(inputs[k])

        # Second pass: Apply transforms (to_tensor, color_aug)
        # Iterate over all keys, including newly created scale 0 keys
        for k in list(inputs.keys()):
            f = inputs[k]
            if "color" in k[0] and k[2] == 0: # Process only scale 0 color images
                name, frame_id, scale, cam_side = k
                inputs[(name, frame_id, scale, cam_side)] = self.to_tensor(f)
                inputs[(name + "_aug", frame_id, scale, cam_side)] = self.to_tensor(self.color_aug(f))
            elif "bev_gt" in k[0] and k[2] == 0:
                # Apply only ToTensor to BEV ground truth
                inputs[k] = self.to_tensor(f)
            elif any(x in k[0] for x in labels_list) and k[2] == 0: # Process only scale 0 labels
                name, frame_id, scale, cam_side = k
                if name == "semantic_labels":
                    inputs[(name, frame_id, scale, cam_side)] = self.to_tensor_semantic_label(np.array(f))
                elif name == "motion_labels":
                    inputs[(name, frame_id, scale, cam_side)] = self.to_tensor_motion_label(np.array(f))
            elif k[0] == "detection_labels": # Detection labels are already tensors, just ensure they are in inputs
                pass # No action needed, they are already processed in get_detection_label
            elif isinstance(f, np.ndarray): # For K, D, displacement_magnitude, theta_lut, angle_lut
                inputs[k] = torch.from_numpy(f)
            # else: it's already a tensor or other type, no action needed.

    def destruct_original_image_tensors(self, inputs):
        # Iterate over a copy of keys to avoid modification issues
        for k in list(inputs.keys()):
            if len(k) > 2 and k[2] == -1:
                del inputs[k]

    def create_and_process_training_items(self, frame_index):
        inputs = dict()
        do_color_aug = self.is_train and random.random() > 0.5

        # Load BEV ground truth image for the frame
        inputs[("bev_gt", 0, -1)] = self.get_bev_image(frame_index)

        # Loop through all 4 camera sides for the current frame_index
        for cam_side in self.all_cam_sides:
            if self.crop:
                if int(frame_index[1:]) < self.total_car1_images:
                    cropped_coords = self.cropped_coords["Car1"][cam_side]
                else:
                    cropped_coords = self.cropped_coords["Car2"][cam_side]
            else:
                cropped_coords = None

            for i in self.frame_idxs:
                # Update keys to include cam_side for all data types
                inputs[("color", i, -1, cam_side)] = self.get_image(i, cropped_coords, frame_index, cam_side)

                inputs[("K_mat", i, cam_side)] = self.get_intrinsic_matrix(cropped_coords, frame_index, cam_side)
                inputs[("extrinsics", i, cam_side)] = self.get_extrinsics(frame_index, cam_side, i)

                if "distance" in self.task or "depth" in self.task:
                    if self.is_train:
                        inputs[("K", i, cam_side)], inputs[("D", i, cam_side)], intrinsics = self.get_intrinsics(cropped_coords,
                                                                                                                 frame_index, cam_side)
                        k1 = intrinsics[4]
                        inputs[("theta_lut", i, cam_side)] = self.LUTs[k1]["theta"]
                        inputs[("angle_lut", i, cam_side)] = self.LUTs[k1]["angle_maps"]

            if "distance" in self.task or "depth" in self.task:
                # Assuming displacement is per-camera, or we take one for the frame and store per-camera for consistency
                inputs[("displacement_magnitude", -1, cam_side)] = self.get_displacements_from_speed(frame_index, cam_side)

            if "semantic" in self.task:
                inputs[("semantic_labels", 0, -1, cam_side)] = self.get_label("semantic_annotations", cropped_coords, frame_index,
                                                                            cam_side)

            if "motion" in self.task:
                inputs[("motion_labels", 0, -1, cam_side)] = self.get_label("motion_annotations", cropped_coords, frame_index,
                                                                          cam_side)

            if "detection" in self.task:
                det_labels = self.get_detection_label(cropped_coords, frame_index, cam_side)
                det_labels[:, 1] = self.all_cam_sides.index(cam_side)  # Add camera index
                inputs[("detection_labels", 0, cam_side)] = det_labels

        if do_color_aug:
            self.color_aug = transforms.ColorJitter(brightness=(0.8, 1.2),
                                                    contrast=(0.8, 1.2),
                                                    saturation=(0.8, 1.2),
                                                    hue=(-0.1, 0.1))
        else:
            self.color_aug = (lambda x: x)

        self.preprocess(inputs)
        self.destruct_original_image_tensors(inputs)

        return inputs

    def __len__(self):
        # The length is now the number of unique frames, as each __getitem__ call processes all cameras for one frame
        return len(self.unique_frame_indices)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.
        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:
            ("color",          <frame_id>, <scale>, <cam_side>)       raw color images,
            ("bev_gt",         <frame_id>, <scale>)                   BEV ground truth image,
            ("K_mat",          <frame_id>, <cam_side>)                camera intrinsics (3x3 matrix),
            ("extrinsics",     <frame_id>, <cam_side>)                camera extrinsics (4x4 matrix),
            ("K",              <frame_id>, <cam_side>)                camera intrinsics,
            ("D",              <frame_id>, <cam_side>)                distortion coefficients,
            ("angle_lut",      <frame_id>, <cam_side>)                look up table containing coords for angle of incidence,
            ("theta_lut",      <frame_id>, <cam_side>)                look up table containing coords for angle in the image plane,
            ("color_aug",      <frame_id>, <scale>, <cam_side>)       augmented color image list similar to above raw color list,
            ("displacement_magnitude", -1, <cam_side>)                displacement from t-1 to t (reference frame)
            ("motion_labels",  <frame_id>, <scale>, <cam_side>)       motion segmentation labels of t (reference frame)
            ("semantic_labels",<frame_id>, <scale>, <cam_side>)       semantic segmentation labels of t (reference frame)
            ("detection_labels", <frame_id>, <cam_side>)              detection labels of t (reference frame)

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',

        <scale> is an integer representing the scale of the image relative to the full size image:
           -1       images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        # Get the unique frame index for this item
        frame_index = self.unique_frame_indices[index]
        return self.create_and_process_training_items(frame_index)

    def collate_fn(self, batch):
        """Collates a batch of data from __getitem__ into a single batch dictionary.
        :param batch: output returned from __getitem__ function
        :return: return modified version from the batch after edit "detection_label"
        """
        if not batch:
            return {}

        collated_batch = {}
        # Get all keys from the first dictionary in the batch
        all_keys = list(batch[0].keys())

        all_detection_labels = []

        for key in all_keys:
            # Skip detection labels for now, handle them at the end
            if key[0] == "detection_labels":
                continue

            tensors_to_stack = [item[key] for item in batch]
            collated_batch[key] = torch.stack(tensors_to_stack, 0)

        # Now handle detection labels by concatenating them all into one tensor
        for i, item in enumerate(batch):
            for key, det_tensor in item.items():
                if key[0] == "detection_labels":
                    det_tensor[:, 0] = i  # Overwrite first column with batch index
                    all_detection_labels.append(det_tensor)

        if all_detection_labels:
            collated_batch[("detection_labels", 0)] = torch.cat(all_detection_labels, 0)

        return collated_batch


