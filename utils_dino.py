import cv2
import os
import numpy as np
import supervision as sv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from supervision.annotators.utils import ColorLookup

import torch
import torchvision

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# Building SAM Model and SAM Predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

################# WALLS UTILS #################

def filter_small_masks(masks, boxes, image_shape, min_area_ratio=0.1):
    """
    Remove masks that cover less than min_area_ratio (e.g., 10%) of the image area.
    
    masks: list of binary np.arrays
    boxes: np.array (N, 4)
    image_shape: tuple (H, W)
    min_area_ratio: float, e.g., 0.1 for 10%
    """
    H, W = image_shape
    image_area = H * W
    keep = []

    for i, mask in enumerate(masks):
        mask_area = np.sum(mask.astype(bool))
        if mask_area / image_area >= min_area_ratio:
            keep.append(i)

    filtered_masks = [masks[i] for i in keep]
    filtered_boxes = boxes[keep]

    return filtered_masks, filtered_boxes

def filter_noisy_masks(masks, boxes, max_components=30, min_component_area=5):
    """
    Removes masks with too many isolated components (small blobs).

    masks: list of binary masks
    boxes: np.array (N, 4)
    max_components: maximum allowed number of connected components 
    min_component_area: ignore very tiny components (likely noise)
    """
    clean_masks = []
    clean_boxes = []

    for mask, box in zip(masks, boxes):
        # Convert to uint8
        mask_uint8 = (mask.astype(np.uint8) * 255)

        # Get connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        
        # Count components above area threshold
        valid_components = sum(stat[cv2.CC_STAT_AREA] > min_component_area for stat in stats[1:])  # skip background

        # print("valid_components", valid_components)

        if valid_components <= max_components:
            # pass
            clean_masks.append(mask)
            clean_boxes.append(box)

    return clean_masks, np.array(clean_boxes)


def remove_masks_that_contain_others(masks, boxes, threshold=0.9):
    """
    masks: List of binary numpy arrays (H, W)
    boxes: np.array of shape (N, 4), bbox coordinates
    threshold: overlap ratio threshold to consider one mask inside another
    """
    num_masks = len(masks)
    keep = np.ones(num_masks, dtype=bool)

    for i in range(num_masks):
        for j in range(num_masks):
            if i == j or not keep[i] or not keep[j]:
                continue

            mask_i = masks[i].astype(bool)
            mask_j = masks[j].astype(bool)

            # Area of each mask
            area_i = np.sum(mask_i)
            area_j = np.sum(mask_j)

            # Overlap area
            intersection = np.logical_and(mask_i, mask_j).sum()

            # Check if j is mostly inside i and i is larger
            if intersection / area_j > threshold and area_i > area_j:
                keep[i] = False  # Remove the larger container mask

    # Filter masks and boxes
    filtered_masks = [m for k, m in zip(keep, masks) if k]
    filtered_boxes = boxes[keep]

    return filtered_masks, filtered_boxes


def sample_random_points_from_mask(mask, num_points=1, seed=None):
    """
    Randomly sample points from a binary mask.
    
    Parameters:
        mask (np.ndarray): Binary mask of shape (H, W)
        num_points (int): Number of points to sample
        seed (int, optional): Seed for reproducibility

    Returns:
        List of (x, y) tuples for sampled points
    """
    if seed is not None:
        np.random.seed(seed)

    # Get all coordinates where mask is True (i.e., 1 or non-zero)
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return []  # No valid points in the mask

    indices = np.random.choice(len(xs), size=min(num_points, len(xs)), replace=False)
    sampled_points = [(xs[i], ys[i]) for i in indices]

    return sampled_points


def get_wall_bboxes_points(SOURCE_IMAGE_PATH):

    BOX_THRESHOLD = 0.25
    TEXT_THRESHOLD = 0.25
    NMS_THRESHOLD = 0.8

    CLASSES = ["walls."]

    print("Getting wall mask from ", SOURCE_IMAGE_PATH)

    if os.path.exists(SOURCE_IMAGE_PATH):
        
        ######################### DO GROUNDING DETECTION ##########################
        # load image
        image = cv2.imread(SOURCE_IMAGE_PATH)

        # detect objects
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        # # annotate image with detections
        # box_annotator = sv.BoxAnnotator(color_lookup = ColorLookup.INDEX)
        # labels = [
        #     f"{CLASSES[class_id]} {confidence:0.2f}" 
        #     for _, _, confidence, class_id, _, _ 
        #     in detections]
        
        # print("labels", labels)
        # print(f"Before NMS: {len(detections.xyxy)} boxes")
        # annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)#, labels=labels)
        # plt.imshow(annotated_frame)
        # plt.show()

        # print("torch.from_numpy(detections.xyxy)", torch.from_numpy(detections.xyxy).dtype)
        # print("torch.from_numpy(detections.confidence)", torch.from_numpy(detections.confidence).dtype)
        
        ############################################################################
        ######################### DO NMS POSTPROCESSING ##########################
        # NMS post process
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            NMS_THRESHOLD
        ).numpy().tolist()


        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        # print(f"After NMS: {len(detections.xyxy)} boxes")

        # box_annotator = sv.BoxAnnotator(color_lookup = ColorLookup.INDEX)
        # annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)
        # plt.imshow(annotated_frame)
        # plt.show()
        
        #############################################################################
        ######################### DO SAM SEGMENTATION ###############################

        # Prompting SAM with detected boxes
        def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
            sam_predictor.set_image(image)
            result_masks = []
            for box in xyxy:
                masks, scores, logits = sam_predictor.predict(
                    box=box,
                    multimask_output=True
                )
                index = np.argmax(scores)
                result_masks.append(masks[index])
            return np.array(result_masks)


        # convert detections to masks
        detections.mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        # print("ALL SAM Segmentations from GroundingDINO BBOX")
        # box_annotator = sv.BoxAnnotator(color_lookup = ColorLookup.INDEX)
        # mask_annotator = sv.MaskAnnotator(color_lookup = ColorLookup.INDEX)
        # annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        # annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)#, labels=labels)
        # plt.imshow(annotated_image)
        # plt.show()

        ################################################################################################
        # FILTER MASKS #
        ################################################################################################
        # First remove small masks (< 10% image area)
        filtered_masks, filtered_boxes = filter_small_masks(
                                                            masks=detections.mask,
                                                            boxes=detections.xyxy,
                                                            image_shape=image.shape[:2],  # (H, W)
                                                            min_area_ratio=0.01
                                                        )

        filtered_masks, filtered_boxes = filter_noisy_masks(filtered_masks,
                                                            filtered_boxes,
                                                            max_components=20,
                                                            min_component_area=10)


        final_masks, final_boxes = remove_masks_that_contain_others(
                                                                    masks=filtered_masks,
                                                                    boxes=filtered_boxes,
                                                                    threshold=0.9
                                                                    )
        ################################################################################################
        ################################################################################################

        detections.mask = final_masks
        detections.xyxy = final_boxes

        # print("FILTERED SAM Segmentations from GroundingDINO BBOX")
        # annotate image with detections
        # box_annotator = sv.BoxAnnotator(color_lookup = ColorLookup.INDEX)
        # mask_annotator = sv.MaskAnnotator(color_lookup = ColorLookup.INDEX)
        # labels = [
        #     f"{CLASSES[class_id]} {confidence:0.2f}" 
        #     for _, _, confidence, class_id, _, _ 
        #     in detections]
        # annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        # annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)#, labels=labels)
        # print(labels)
        # plt.imshow(annotated_image)
        # plt.show()


        ################################################################################################
        ################################################################################################
        selected_points = []
        for i, mask in enumerate(final_masks):  # mask: (H, W) binary
            mask_uint8 = mask.astype(np.uint8)

            # Compute center from moments
            moments = cv2.moments(mask_uint8)
            if moments["m00"] != 0:
                original_cx = int(moments["m10"] / moments["m00"])
                original_cy = int(moments["m01"] / moments["m00"])
            else:
                ys, xs = np.where(mask)
                original_cx = int(np.mean(xs))
                original_cy = int(np.mean(ys))

            cx, cy = original_cx, original_cy

            if not mask[cy, cx]:
                ys, xs = np.where(mask)
                distances = np.sqrt((xs - cx)**2 + (ys - cy)**2)

                # Get index of 25th percentile closest point
                percentile_index = int(len(distances) * 0.2)
                sorted_indices = np.argsort(distances)

                idx = sorted_indices[percentile_index]
                cx, cy = int(xs[idx]), int(ys[idx])

            # Draw center (green)
            # cv2.circle(annotated_image, (cx, cy), radius=5, color=(0, 255, 0), thickness=-1)

            selected_points.append([cx, cy])

            # Draw original center (red)
            # cv2.circle(annotated_image, (original_cx, original_cy), radius=4, color=(0, 0, 255), thickness=2)

            # Draw line (blue)
            # cv2.line(annotated_image, (original_cx, original_cy), (cx, cy), color=(255, 0, 0), thickness=1)
        ################################################################################################
        ################################################################################################


        # save the annotated grounded-sam image
        # cv2.imwrite(os.path.join("./output/BEST_wall_masks/", f"{img[:-4]}_wall.jpg"), annotated_image)
        # print("WITH BBOX CENTER POINTS")

        # print("walls_bbox", detections.xyxy)
        # print("selected_wall_points", selected_points)
        
        # plt.imshow(annotated_image)
        # plt.show()
        
        # for idx, mask in enumerate(detections.mask):
        #     mask_uint8 = (mask.astype(np.uint8)) * 255
        #     cv2.imwrite(os.path.join("./output/BEST_wall_masks/", f"{img[:-4]}_{idx}_wall_mask.jpg"), mask_uint8)

        return detections.xyxy, selected_points
    

################# FLOOR UTILS #################

def get_rug_masks(SOURCE_IMAGE_PATH):

    BOX_THRESHOLD  = 0.25
    TEXT_THRESHOLD = 0.25
    NMS_THRESHOLD  = 0.8

    CLASSES = ["rug"]

    # load image
    image = cv2.imread(SOURCE_IMAGE_PATH)

    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )
    # box_annotator = sv.BoxAnnotator()
    # annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)#, labels=labels)

    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        NMS_THRESHOLD
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    # Prompting SAM with detected boxes
    def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )

    # annotate image with detections
    # box_annotator = sv.BoxAnnotator()
    # mask_annotator = sv.MaskAnnotator()
    # labels = [
    #     f"{CLASSES[class_id]} {confidence:0.2f}" 
    #     for _, _, confidence, class_id, _, _ 
    #     in detections]
    # annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    # annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)

    # plt.imshow(annotated_image)
    # plt.show()

    all_rug_labels = [
        (CLASSES[class_id],confidence) 
        for _, _, confidence, class_id, _, _ 
        in detections]
    all_rug_bbox = detections.xyxy
    all_rug_masks = detections.mask
    
    # print("RUG Labels", all_rug_labels)
    # print("RUG BBOX", all_rug_bbox)
    # print("RUG Masks", all_rug_masks)

    return all_rug_labels, all_rug_bbox, all_rug_masks
    
def get_floor_masks(SOURCE_IMAGE_PATH):

    BOX_THRESHOLD  = 0.25
    TEXT_THRESHOLD = 0.25
    NMS_THRESHOLD  = 0.8

    CLASSES = ["floor"]

    # load image
    image = cv2.imread(SOURCE_IMAGE_PATH)

    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )
    # box_annotator = sv.BoxAnnotator()
    # annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)#, labels=labels)

    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        NMS_THRESHOLD
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    # Prompting SAM with detected boxes
    def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )

    # annotate image with detections
    # box_annotator = sv.BoxAnnotator()
    # mask_annotator = sv.MaskAnnotator()
    # labels = [
    #     f"{CLASSES[class_id]} {confidence:0.2f}" 
    #     for _, _, confidence, class_id, _, _ 
    #     in detections]
    # annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    # annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)

    # plt.imshow(annotated_image)
    # plt.show()

    all_floor_labels = [
        (CLASSES[class_id],confidence) 
        for _, _, confidence, class_id, _, _ 
        in detections]
    all_floor_bbox = detections.xyxy
    all_floor_masks = detections.mask
    
    # print("FLOOR Labels", all_floor_labels)
    # print("FLOOR BBOX",   all_floor_bbox)
    # print("FLOOR Mask",   all_floor_masks)

    return all_floor_labels, all_floor_bbox, all_floor_masks


def filter_floor_masks(all_floor_labels, all_floor_bbox, floor_masks):
    """
    floor_masks: List of binary masks (numpy arrays of same shape)
    Returns: Filtered list of masks based on overlap and size
    """
    if len(floor_masks) <= 1:
        return all_floor_labels, all_floor_bbox, floor_masks

    kept_indices = []
    used_indices = set()

    for i in range(len(floor_masks)):
        if i in used_indices:
            continue
        mask_i = floor_masks[i]
        keep = True
        for j in range(i + 1, len(floor_masks)):
            if j in used_indices:
                continue
            mask_j = floor_masks[j]

            # Check for overlap
            overlap = np.logical_and(mask_i, mask_j)
            if np.any(overlap):
                area_i = np.sum(mask_i)
                area_j = np.sum(mask_j)
                if area_i >= area_j:
                    used_indices.add(j)  # Discard j
                else:
                    keep = False
                    used_indices.add(i)  # Discard i
                    break
        if keep:
            kept_indices.append(i)

    # Add any masks not used yet (non-overlapping)
    for k in range(len(floor_masks)):
        if k not in used_indices and k not in kept_indices:
            kept_indices.append(k)

    filtered_floor_labels = [all_floor_labels[i] for i in kept_indices]
    filtered_floor_bbox   = [all_floor_bbox[i] for i in kept_indices]
    filtered_floor_masks  = [floor_masks[i] for i in kept_indices]

    return filtered_floor_labels, filtered_floor_bbox, filtered_floor_masks


def overlay_masks_and_bboxes(image, masks, bboxes=None, alpha=0.5, colors=None, selected_floor_points=None):
    """
    image: HxWx3 RGB image
    masks: list of HxW binary masks
    bboxes: list of bounding boxes in [x1, y1, x2, y2] format (same order as masks)
    alpha: transparency for mask overlay
    colors: list of colors for masks and boxes
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    if colors is None:
        colors = plt.cm.get_cmap('tab10', len(masks)).colors

    ax = plt.gca()

    for idx, mask in enumerate(masks):
        color = colors[idx % len(colors)]
        colored_mask = np.zeros_like(image, dtype=np.float32)
        for c in range(3):
            colored_mask[..., c] = mask * color[c]

        blended = image / 255.0 * (1 - alpha) + colored_mask * alpha
        plt.imshow(blended)

        # Draw bounding box if provided
        # if bboxes and (idx < len(bboxes)):
        x1, y1, x2, y2 = bboxes[idx]
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)

    if selected_floor_points:
        for point in selected_floor_points:
            plt.scatter(point[1], point[0])

    # plt.axis('off')
    # plt.title("Image with Masks and Bounding Boxes")
    plt.show()



def get_10th_percentile_true_point(mask):
    coords = np.argwhere(mask)

    if coords.shape[0] == 0:
        return None  # No True values

    centroid = coords.mean(axis=0)
    distances = np.linalg.norm(coords - centroid, axis=1)

    # Compute 10th percentile distance
    percentile_threshold = np.percentile(distances, 10)

    # Get indices where distance is closest to the 10th percentile
    idx = np.argmin(np.abs(distances - percentile_threshold))
    return list(coords[idx])  # (row, col)

def filter_rugs_with_floor_update_and_revert(
    floor_labels, floor_bboxes, floor_masks,
    rug_labels, rug_bboxes, rug_masks,
    floor_removal_threshold=0.05  # fraction of floor area below which we revert
):
    """
    Filters rugs by overlap or bbox inclusion.
    Updates floor masks by subtracting rug areas only if floor not almost fully removed.
    If floor almost fully removed, revert mask and discard that rug.

    Returns:
        floor_labels,
        updated_floor_bboxes,
        updated_floor_masks,
        valid_rug_labels,
        valid_rug_bboxes,
        valid_rug_masks,
        rug_to_floor_indices
    """

    updated_floor_masks = [mask.copy() for mask in floor_masks]
    valid_rug_labels = []
    valid_rug_bboxes = []
    valid_rug_masks = []
    rug_to_floor_indices = []

    for i, (rug_mask, rug_bbox) in enumerate(zip(rug_masks, rug_bboxes)):
        print("RUG: ", i)
        overlapped_floors = []
        rug_valid = False

        # Track floors that overlapped and whether all survived subtraction
        floors_to_update = []

        # 1) Check overlaps with floors
        for j, floor_mask in enumerate(updated_floor_masks):
            overlap = np.logical_and(rug_mask, floor_mask)
            if np.any(overlap):
                print("There is overlap")
                floors_to_update.append(j)

        # 2) If overlaps exist, try subtraction but revert if floor nearly gone
        if floors_to_update:
            can_keep = True
            for j in floors_to_update:
                original_floor_mask = updated_floor_masks[j].copy()
                subtracted_mask = np.logical_and(updated_floor_masks[j], np.logical_not(rug_mask))

                # Calculate area fraction left after subtraction
                original_area = np.sum(original_floor_mask)
                remaining_area = np.sum(subtracted_mask)
                if original_area == 0:
                    # Avoid division by zero, treat as no floor
                    can_keep = False
                    break
                fraction_left = remaining_area / original_area

                if fraction_left < floor_removal_threshold:
                    print("Subtract rug from floor area NOT OK. Revert to original Floor!")
                    # Revert all floor masks updated so far and discard rug
                    for revert_j in floors_to_update:
                        updated_floor_masks[revert_j] = floor_masks[revert_j].copy()
                    can_keep = False
                    break
                else:
                    # Update floor mask with subtraction
                    print("Subtract rug from floor area OK. Floor updated!")
                    updated_floor_masks[j] = subtracted_mask

            if can_keep:
                rug_valid = True
                overlapped_floors = floors_to_update

        # 3) If no overlaps, check bbox inside floors
        # if not rug_valid:
        else:
            print("No overlap")
            
            rug_x1, rug_y1, rug_x2, rug_y2 = rug_bbox
            for j, floor_bbox in enumerate(floor_bboxes):
                floor_x1, floor_y1, floor_x2, floor_y2 = floor_bbox
                bbox_inside = (
                    rug_x1 >= floor_x1-1 and rug_y1 >= floor_y1-1 and
                    rug_x2 <= floor_x2+1 and rug_y2 <= floor_y2+1
                )
                print("RUG BBOX", rug_x1, rug_y1, rug_x2, rug_y2)
                print("FLOOR BBOX", floor_x1-1, floor_y1-1, floor_x2+1, floor_y2+1)
                if bbox_inside:
                    print("YES bbox is inside Floor area")
                    rug_valid = True
                    overlapped_floors = [j]
                    break

        # 4) Collect valid rugs info
        if rug_valid:
            valid_rug_labels.append(rug_labels[i])
            valid_rug_bboxes.append(rug_bbox)
            valid_rug_masks.append(rug_mask)
            rug_to_floor_indices.append(overlapped_floors)

    # 5) Recompute updated floor bboxes
    # updated_floor_bboxes = []
    # for mask in updated_floor_masks:
    #     ys, xs = np.where(mask)
    #     if len(xs) > 0 and len(ys) > 0:
    #         x1, y1, x2, y2 = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
    #     else:
    #         x1 = y1 = x2 = y2 = 0
    #     updated_floor_bboxes.append([x1, y1, x2, y2])

    # print("updated_floor_masks.shape", updated_floor_masks[0].shape)

    selected_floor_points = []
    for mask in updated_floor_masks:
        point = get_10th_percentile_true_point(mask)
        selected_floor_points.append(point)

    return (
        floor_labels,
        floor_bboxes,
        floor_masks,
        valid_rug_labels,
        valid_rug_bboxes,
        valid_rug_masks,
        rug_to_floor_indices,
        selected_floor_points
    )


def get_floor_bboxes_points_with_rug(SOURCE_IMAGE_PATH):
    
    all_rug_labels, all_rug_bbox, all_rug_masks = get_rug_masks(SOURCE_IMAGE_PATH)
    all_floor_labels, all_floor_bbox, all_floor_masks = get_floor_masks(SOURCE_IMAGE_PATH)

    # image = cv2.imread(SOURCE_IMAGE_PATH)

    filtered_floor_labels, filtered_floor_bbox, filtered_floor_masks = filter_floor_masks(all_floor_labels, all_floor_bbox, all_floor_masks)

    (
        updated_floor_labels,
        updated_floor_bboxes,
        updated_floor_masks,
        valid_rug_labels,
        valid_rug_bboxes,
        valid_rug_masks,
        rug_to_floor_indices,
        selected_floor_points
    ) = filter_rugs_with_floor_update_and_revert(
        filtered_floor_labels,
        filtered_floor_bbox,
        filtered_floor_masks,
        all_rug_labels,
        all_rug_bbox,
        all_rug_masks,
        floor_removal_threshold=0.05  # 5% floor area left minimum
    )
        
    # print("len(valid_rug_masks)", len(valid_rug_masks))
    # overlay_masks_and_bboxes(image, valid_rug_masks, valid_rug_bboxes)

    # print("len(updated_floor_masks)", len(updated_floor_masks))
    # overlay_masks_and_bboxes(image, filtered_floor_masks, filtered_floor_bbox, selected_floor_points=selected_floor_points)

    # print("rug_to_floor_indices", rug_to_floor_indices)

    return np.array(filtered_floor_bbox), selected_floor_points, np.array(valid_rug_bboxes), rug_to_floor_indices



############ ONNX UTILS ##############

def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   