# 02102025
# extra code for door detection

def is_fully_visible(door_mask, door_bbox, threshold=0.8):
    x1, y1, x2, y2 = map(int, door_bbox)  # convert bbox coords to int
    bbox_area = (x2 - x1) * (y2 - y1)
    
    # Mask area inside bbox
    mask_area = np.sum(door_mask[y1:y2, x1:x2] > 0)
    
    visible_ratio = mask_area / bbox_area
    return visible_ratio >= threshold
  
def is_quadrilateral(
    mask: np.ndarray,
    epsilon_ratio: float = 0.02,
    coverage_thresh: float = 0.85
    ) -> bool:
    """
    Check if a mask is roughly a quadrilateral (door-like)
    and that the contour area covers most of the mask.

    Args:
        mask: binary or boolean mask of the object
        epsilon_ratio: approximation accuracy (fraction of contour perimeter)
        corner_tol: tolerance for corner count (± this value)
        coverage_thresh: minimum ratio of contour_area / mask_area
    """
    mask_uint8 = mask.astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    # Take largest contour (door mask)
    contour = max(contours, key=cv2.contourArea)

    # Polygon approximation
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon_ratio * peri, True)

    # Check number of corners
    n_corners = len(approx)
    print("Number of corners detected:", n_corners)

    # --- Coverage check ---
    contour_area = cv2.contourArea(contour)
    mask_area = np.sum(mask_uint8 > 0)
    coverage = contour_area / float(mask_area + 1e-6)
    print(f"Coverage ratio: {coverage:.2f}")

    return n_corners == 4 , (coverage >= coverage_thresh)

def show_contour(image: np.ndarray, mask: np.ndarray, epsilon_ratio: float = 0.02):
    """
    Draw the largest contour and its polygonal approximation on the image.
    
    Args:
        image: RGB image (H, W, 3)
        mask: boolean or binary mask (H, W)
        epsilon_ratio: contour approximation factor
    """
    mask_uint8 = mask.astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contour found")
        return image

    # Largest contour
    contour = max(contours, key=cv2.contourArea)

    # Approximate polygon
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon_ratio * peri, True)

    # Copy image for visualization
    vis = image.copy()

    # Draw contour (green)
    cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)

    # Draw approximated polygon (red)
    cv2.drawContours(vis, [approx], -1, (255, 0, 0), 2)

    # Draw corners as circles
    for (x, y) in approx.reshape(-1, 2):
        cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)

    # Show
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    return vis

def get_door_bboxes_points(SOURCE_IMAGE_PATH):

    # 1) DEFALUT DOOR PARAMS + 80% area check = OK !!!
    # BOX_THRESHOLD   = 0.65
    # TEXT_THRESHOLD  = 0.20
    # NMS_THRESHOLD   = 0.8
    # TEXT_PROMPT = "doors"

    # 2) NEW DOOR PARAMS + 80% area check = OK !!!
    BOX_THRESHOLD   = 0.60
    TEXT_THRESHOLD  = 0.80
    NMS_THRESHOLD   = 0.8
    TEXT_PROMPT = "doors"

    if os.path.exists(SOURCE_IMAGE_PATH):
        
        ######################### DO GROUNDING DETECTION ##########################
        # load image
        image = cv2.imread(SOURCE_IMAGE_PATH)

        image_source, image_dino = load_image(SOURCE_IMAGE_PATH)
        boxes, logits, phrases = predict(
            model=grounding_dino_model,
            image=image_dino,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        detections_box = xyxy
        detections_scores = logits.cpu().numpy()
        
        ############################################################################
        ######################### DO NMS POSTPROCESSING ##########################
        # NMS post process
        nms_idx = nms(
            torch.from_numpy(detections_box), 
            torch.from_numpy(detections_scores), 
            NMS_THRESHOLD
        ).numpy().tolist()


        detections_box = detections_box[nms_idx]
        detections_scores = detections_scores[nms_idx]

        if len(detections_box) == 0:
            print("No door detected")
            return [], []

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
        detections_masks = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections_box
        )

        selected_points = []
        for bbox in detections_box:  # each bbox = [x_min, y_min, x_max, y_max]
            x_min, y_min, x_max, y_max = bbox

            # Center of bbox
            cx = int((x_min + x_max) / 2)
            cy = int((y_min + y_max) / 2)

            selected_points.append([cx, cy])

        filtered_masks = []
        filtered_bboxes = []
        filtered_points = []

        for mask, bbox, point in zip(detections_masks, detections_box, selected_points):
            
            if is_fully_visible(mask, bbox, threshold=0.80):
                print("✅ Door Mask is fully visible")
                is_quadrilateral_flag, coverage_flag = is_quadrilateral(mask)
                if is_quadrilateral_flag:
                    print("✅ Door Mask is quadrilateral")
                    if coverage_flag:
                        print("✅ Door Mask passed coverage check")
                        filtered_masks.append(mask)
                        filtered_bboxes.append(bbox)
                        filtered_points.append(point)
                # show_contour(image, mask, epsilon_ratio=0.02)

        return np.array(filtered_bboxes), filtered_points
    
##########################################################################3
# add this to Image2Scene.get_bboxes_points
door_bboxes, door_selected_points = get_door_bboxes_points(image_path)

# new output with added door info
output = {
    'wall_bboxes'          : wall_bboxes.tolist(),
    'wall_selected_points' : wall_selected_points,
    'floor_bboxes'         : floor_bboxes.tolist(),
    'floor_selected_points': floor_selected_points,
    'rug_bboxes'           : rug_bboxes.tolist(),
    'rug_to_floor_indices' : rug_to_floor_indices,
    'door_bboxes'          : door_bboxes.tolist(),
    'door_selected_points' : door_selected_points
    }
##########################################################################