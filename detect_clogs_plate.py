import json
import cv2
import numpy as np
from pupil_apriltags import Detector as apriltag 
import matplotlib.pyplot as plt

def threshold_video_movement(video_path: str, roi=None) -> list[dict]:
    """
    Detect significant changes in a video using background subtraction and thresholding.

    Args:
        video_path: Path to the video file.
        roi: Region of interest as (x, y, width, height). If None, entire frame is used.

    Returns:
        A list of dictionaries, each containing:
        - 'thresholded_image': the thresholded image for the cycle
        - 'fiducial_coordinate': the coordinate of the fiducial (if any)
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Create a background subtractor object
    backSub = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=40, detectShadows=False)
    
    # Initialize variables
    cycles = []
    current_cycle = None
    state = 'waiting_for_cycle'
    fiducial_coordinates = None
    i = 0
    staleness = 0
    

    if roi is None:
        try:
            with open("roi_info.json", "r") as f:
                roi_data = json.load(f)
                roi = roi_data.get("roi", None)
                print(f"Loaded saved ROI: {roi}")
        except FileNotFoundError:
            print("No saved ROI found, asking user to select one.")
        
        if roi is None:
            ret, first_frame = cap.read()
            if ret:
                roi = select_roi(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        i += 1
        ret, frame = cap.read()
        if not ret:
            break  # End of video
            
        if roi is not None:
            x, y, w, h = roi
            h_img, w_img = frame.shape[:2]
            end_x = min(x + w, w_img)
            end_y = min(y + h, h_img)
            if x < w_img and y < h_img:
                roi_frame = frame[y:end_y, x:end_x]
                frame = roi_frame 
        
        if fiducial_coordinates is None:
            fiducial_coordinates = detect_fiducials(frame)
            if len(fiducial_coordinates) == 0:
                # No fiducials detected yet, head not yet in frame
                fiducial_coordinates = None
                continue
            elif roi is not None:
                adjusted_coordinates = {}
                for key, coord in fiducial_coordinates.items():
                    adjusted_coordinates[key] = type('obj', (object,), {
                        'x': coord[0] - x,
                        'y': coord[1] - y
                    })
                fiducial_coordinates = adjusted_coordinates

        # Apply background subtraction to get the foreground mask
        fg_mask = backSub.apply(frame)

        movement_amount = np.count_nonzero(fg_mask)

        # Define movement thresholds
        low_movement_threshold = 0.025 * fg_mask.size
        high_movement_threshold = 0.05 * fg_mask.size

        if state == 'waiting_for_cycle':
            if movement_amount < low_movement_threshold:
                # Start a new cycle
                current_cycle = {'accum_mask': np.zeros_like(fg_mask, dtype=np.float32),
                                 'frame_count': 0}
                state = 'in_cycle'
                print("Starting new cycle")
        elif state == 'in_cycle':
            # Accumulate the fg_mask
            current_cycle['accum_mask'] += fg_mask.astype(np.float32)
            current_cycle['frame_count'] += 1

            if movement_amount > high_movement_threshold:
                # End of cycle
                cycles.append(current_cycle)
                current_cycle = None
                state = 'waiting_for_movement'
                print("End of cycle")
        elif state == 'waiting_for_movement':
            if movement_amount < low_movement_threshold:
                state = 'between_cycles'
                print("Between cycles")
        elif state == 'between_cycles':
            if movement_amount > high_movement_threshold and staleness > 10:
                # Robot moving back into frame
                print(f'Staleness: {staleness}')
                staleness = 0
                state = 'waiting_for_cycle'
                print("Waiting for cycle")
            else:
                staleness += 1
                if staleness > 50:
                    print("Staleness limit reached")
                    # 50 frames of no movement, the robot isn't coming back for a second cycle at this point
                    # This might need to be adjusted
                    break

    # After processing all frames
    if current_cycle is not None:
        cycles.append(current_cycle)

    cap.release()

    # For each cycle, normalize the accumulated mask and process
    for cycle in cycles:
        frame_count = cycle['frame_count']
        accumulated_mask = cycle['accum_mask']
        if frame_count > 0:
            accumulated_mask /= frame_count

        # Convert accumulated mask to 8-bit image
        accumulated_mask = np.nan_to_num(accumulated_mask)
        accum_mask_uint8 = cv2.normalize(accumulated_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply thresholding
        _, thresholded_image = cv2.threshold(accum_mask_uint8, 0, 255, cv2.THRESH_BINARY)

        cycle['thresholded_image'] = thresholded_image
        cycle['fiducial_coordinates'] = fiducial_coordinates
        if roi is not None:
            cycle['roi'] = roi

    if roi is not None:
        try:
            with open('roi_info.json', 'w') as f:
                json.dump({'roi': roi}, f, indent=4)
        except Exception as e:
            print(f"Save ROI Info Error: {e}")

    return cycles

def select_roi(frame):
    """Choose ROI"""
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk
    
    SCALING_FACTOR = 1
    
    roi_window = tk.Tk()
    roi_window.title("Choose Region of Interest")
    
    roi_data = {'start_x': 0, 'start_y': 0, 'end_x': 0, 'end_y': 0, 'confirmed': False}
    
    img_pil = Image.fromarray(frame)
    img_width = img_pil.width // SCALING_FACTOR
    img_height = img_pil.height // SCALING_FACTOR
    img_pil = img_pil.resize((img_width, img_height))
    img_tk = ImageTk.PhotoImage(img_pil)
    
    canvas = tk.Canvas(roi_window, width=img_width, height=img_height)
    canvas.pack(padx=10, pady=10)
    
    canvas.img_tk = img_tk
    
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    
    rect_id = None
    
    def on_mouse_down(event):
        roi_data['start_x'] = event.x
        roi_data['start_y'] = event.y
        
    def on_mouse_move(event):
        nonlocal rect_id
        if rect_id:
            canvas.delete(rect_id)
        
        roi_data['end_x'] = event.x
        roi_data['end_y'] = event.y
        
        rect_id = canvas.create_rectangle(
            roi_data['start_x'], roi_data['start_y'],
            roi_data['end_x'], roi_data['end_y'],
            outline='red', width=2
        )
        
    def confirm_roi():
        x = min(roi_data['start_x'], roi_data['end_x']) * SCALING_FACTOR
        y = min(roi_data['start_y'], roi_data['end_y']) * SCALING_FACTOR
        w = abs(roi_data['end_x'] - roi_data['start_x']) * SCALING_FACTOR
        h = abs(roi_data['end_y'] - roi_data['start_y']) * SCALING_FACTOR
        
        roi_data['confirmed'] = True
        roi_data['roi'] = (int(x), int(y), int(w), int(h))
        roi_window.destroy()
        
    def use_full_image():
        roi_data['confirmed'] = True
        roi_data['roi'] = None
        roi_window.destroy()
    
    canvas.bind("<ButtonPress-1>", on_mouse_down)
    canvas.bind("<B1-Motion>", on_mouse_move)
    
    btn_frame = ttk.Frame(roi_window)
    btn_frame.pack(pady=10)
    
    confirm_btn = ttk.Button(btn_frame, text="Save", command=confirm_roi)
    confirm_btn.pack(side=tk.LEFT, padx=5)
    
    skip_btn = ttk.Button(btn_frame, text="Whole figure", command=use_full_image)
    skip_btn.pack(side=tk.LEFT, padx=5)
    
    roi_window.mainloop()
    
    if roi_data.get('confirmed', False) and roi_data.get('roi') is not None:
        return roi_data['roi']
    else:
        return None


def detect_fiducials(frame: np.ndarray) -> dict[int, np.ndarray]:
    """
    Detect fiducial in the frame using AprilTag.

    Args:
        frame: Image frame.

    Returns:
        The coordinate of the fiducial.
    """

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = apriltag(families='tag36h11')

    detections = detector.detect(img)
    centers = {}

    for detection in detections:
        centers[detection.tag_id] = tuple(detection.center)
        print(f"Detected tag: {detection.tag_id}, Center: {detection.center}")

    return centers


def homography(fiducial_coordinates: dict[int, tuple[float, float]], image: np.ndarray, row: str) -> np.ndarray:
    """
    Apply homography transformation to the image using the front fiducial coordinate.
    Modified to match the processing in calibration.py.
    
    Args:
        fiducial_coordinate: The coordinate of the front fiducial.
        image: The input image.
        row: The row of nozzles to process ('A' or 'B').
    
    Returns:
        The warped image after homography transformation.
    """
    # Load the absolute coordinates of the nozzles from calibration_data.json
    calibration_data = json.load(open('calibration_data_plate.json'))
    
    # Use the same coordinate extraction logic as in calibration.py
    pts_src = np.array([calibration_data[row]["homography"]], dtype=float)
    
    # Define destination points (same as in calibration.py)
    
    if 'roi' in calibration_data[row]:
        _, _, w, h = calibration_data[row]['roi']
        width, height = w, h
        print(f"ROI from calibration_data[{row}]:", calibration_data[row].get("roi"))

    else:
        width, height = 400, 300
    # width, height = 400, 300
    pts_dst = np.array([
        [0, 0],              # Top-left
        [width - 1, 0],      # Top-right
        [0, height - 1],     # Bottom-left
        [width - 1, height - 1]  # Bottom-right
    ], dtype=float)
    
    # Print debug information
    print(f"Using coordinates for section {row}:")
    print(f"Source points: {pts_src}")
    print(f"Destination points: {pts_dst}")
    
    # Calculate the homography matrix
    homography_matrix, status = cv2.findHomography(pts_src, pts_dst)
    
    # Apply perspective transformation
    warped_image = cv2.warpPerspective(image, homography_matrix, (width, height))
    
    # Convert to RGB for display (calibration.py uses RGB for display)
    if len(warped_image.shape) == 2 or warped_image.shape[2] == 1:
        display_image = cv2.cvtColor(warped_image, cv2.COLOR_GRAY2RGB)
    else:
        display_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)
    
    # Save the image to file for comparison
    cv2.imwrite(f"homography_output_{row}.png", warped_image)
    
    # Display similar to calibration.py
    # plt.figure(figsize=(10, 6))
    # plt.imshow(display_image)
    # plt.title(f"Warped Image for Section {row}")
    # plt.axis('off')
    # plt.show()

    print(f"Warped image shape for section {row}: {warped_image.shape}")
    cv2.imwrite(f"original_section_{row}.png", image)
    cv2.imwrite(f"warped_section_{row}.png", warped_image)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Original Image - Section {row}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Warped Image - Section {row}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    
    return warped_image

def classify_nozzles(warped_image: np.ndarray, section: str = 'B', display: bool = False) -> tuple[dict[str, list[bool]], float]:
    """
    Compute the white ratio for each nozzle and determine if it's clogged.

    - Adds a margin to each nozzle region to avoid including water streams from adjacent nozzle rows.

    Args:
        warped_image: The homography-warped image of the nozzle region.
        section: The nozzle row being processed ('A' or 'B').
        display: Whether to display the result.

    Returns:
        - A dictionary with nozzle clogging status
        - The average white ratio for this row
    """

    num_nozzles = 8
    width = warped_image.shape[1]
    height = warped_image.shape[0]

    nozzle_width = width // num_nozzles
    margin_ratio = 0.20  # Margin width ratio, e.g., 20%

    white_ratios = []

    for i in range(num_nozzles):
        # Compute x-range for each nozzle with margin
        x_start = int(i * nozzle_width + margin_ratio * nozzle_width)  # Left boundary
        x_end = int((i + 1) * nozzle_width - margin_ratio * nozzle_width)  # Right boundary

        # y-range remains fixed
        y_start = int(0.10 * height)
        y_end = int(0.90 * height)
        nozzle_region = warped_image[y_start:y_end, x_start:x_end]

        # Convert to grayscale if needed
        if len(nozzle_region.shape) == 3:
            nozzle_region = cv2.cvtColor(nozzle_region, cv2.COLOR_BGR2GRAY)

        # Compute white ratio
        white_pixels = cv2.countNonZero(nozzle_region)
        total_pixels = nozzle_region.size
        white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
        white_ratios.append(white_ratio)

    # Compute mean and standard deviation
    mean_ratio = np.mean(white_ratios)
    std_ratio = np.std(white_ratios)
    threshold_z = -1.75  # Z-score threshold

    nozzle_status = []

    for i, ratio in enumerate(white_ratios):
        z_score = (ratio - mean_ratio) / std_ratio if std_ratio > 0 else 0
        is_clogged = z_score < threshold_z or ratio < 0.1
        print(f"Nozzle {i + 1}: White ratio = {ratio:.2f}, Z-score = {z_score:.2f}, Clogged = {is_clogged}")
        nozzle_status.append(is_clogged)

    # Visualization
    if display:
        if len(warped_image.shape) == 2:
            warped_image = cv2.cvtColor(warped_image, cv2.COLOR_GRAY2BGR)

        for i, status in enumerate(nozzle_status):
            x_start = int(i * nozzle_width + margin_ratio * nozzle_width)
            x_end = int((i + 1) * nozzle_width - margin_ratio * nozzle_width)

            y_start = int(0.10 * height)
            y_end = int(0.90 * height)

            color = (0, 0, 255) if status else (0, 255, 0)
            cv2.rectangle(warped_image, (x_start, y_start), (x_end, y_end), color, 2)

        cv2.imwrite(f"nozzle_classification_{section}.png", warped_image)
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Nozzle Classification for Section {section}")
        plt.axis('off')
        plt.show()

    return {section: nozzle_status}, mean_ratio
