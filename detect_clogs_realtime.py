import os
import json
import cv2
import numpy as np
from pupil_apriltags import Detector as apriltag 
import matplotlib.pyplot as plt

# from detect_clogs_realtime import homography, classify_nozzles

def threshold_video_movement_realtime_with_config(
    video_path: str,
    ratio_thresh: float = None,
    initial_frame_num: int = None,
    display: bool = False
) -> dict[str, list[bool]] | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    if (ratio_thresh is None or initial_frame_num is None) and os.path.exists("calibration_config.json"):
        with open("calibration_config.json", "r") as f:
            config = json.load(f)
        ratio_thresh = config.get("best_threshold", 0.10)
        initial_frame_num = config.get("start_frame", 50)
        print(f"Loaded calibration_config.json: ratio_thresh={ratio_thresh}, initial_frame={initial_frame_num}")
    else:
        ratio_thresh = ratio_thresh if ratio_thresh is not None else 0.10
        initial_frame_num = initial_frame_num if initial_frame_num is not None else 50
        print(f"Using default/manual parameters: ratio_thresh={ratio_thresh}, initial_frame={initial_frame_num}")

    backSub = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=40, detectShadows=False)

    current_cycle = None
    state = 'waiting_for_cycle'
    fiducial_coordinates = None
    accumulated_frames = 0
    staleness = 0
    i = 0
    detect_interval = 10

    results = {
        "A": None,
        "B": None
    }

    while True:
        i += 1
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        if fiducial_coordinates is None:
            fiducial_coordinates = detect_fiducials(frame)
            if len(fiducial_coordinates) == 0:
                fiducial_coordinates = None
                continue

        fg_mask = backSub.apply(frame)
        movement_amount = np.count_nonzero(fg_mask)

        low_movement_threshold = 0.025 * fg_mask.size
        high_movement_threshold = 0.05 * fg_mask.size

        if state == 'waiting_for_cycle':
            if movement_amount < low_movement_threshold:
                print("Starting new cycle")
                current_cycle = np.zeros_like(fg_mask, dtype=np.float32)
                accumulated_frames = 0
                state = 'in_cycle'

        elif state == 'in_cycle':
            current_cycle += fg_mask.astype(np.float32)
            accumulated_frames += 1

            if accumulated_frames >= initial_frame_num and (accumulated_frames - initial_frame_num) % detect_interval == 0:
                print("Performing clog classification...")
                accumulated_mask = current_cycle / accumulated_frames
                accum_mask_uint8 = cv2.normalize(accumulated_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                _, thresholded_image = cv2.threshold(accum_mask_uint8, 0, 255, cv2.THRESH_BINARY)

                warped_back = homography(fiducial_coordinates, thresholded_image, "B")
                warped_front = homography(fiducial_coordinates, thresholded_image, "A")

                report_back, mean_ratio_back = classify_nozzles(
                    warped_back, section="B", display=display, ratio_thresh=ratio_thresh)
                report_front, mean_ratio_front = classify_nozzles(
                    warped_front, section="A", display=display, ratio_thresh=ratio_thresh)

                if mean_ratio_front > mean_ratio_back:
                    print("This cycle identified as section A")
                    if any(report_front["A"]):
                        print("Clog detected in A, returning")
                        cap.release()
                        return report_front
                    else:
                        results["A"] = report_front
                else:
                    print("This cycle identified as section B")
                    if any(report_back["B"]):
                        print("Clog detected in B, returning")
                        cap.release()
                        return report_back
                    else:
                        results["B"] = report_back

            if movement_amount > high_movement_threshold:
                print("End of cycle, robot moving")
                current_cycle = None
                accumulated_frames = 0
                state = 'waiting_for_movement'

        elif state == 'waiting_for_movement':
            if movement_amount < low_movement_threshold:
                print("Between cycles")
                state = 'between_cycles'

        elif state == 'between_cycles':
            if movement_amount > high_movement_threshold and staleness > 10:
                print(f"Staleness: {staleness}")
                staleness = 0
                state = 'waiting_for_cycle'
                print("Waiting for new cycle")
            else:
                staleness += 1
                if staleness > 50:
                    print("Staleness limit reached, ending video")
                    break

    cap.release()
    print("No clog detected after entire video. Returning last known A and B results.")
    return results



def detect_fiducials(frame: np.ndarray) -> dict[int, np.ndarray]:
    """
    Detect fiducials in the frame using AprilTag.

    Args:
        frame: Image frame.

    Returns:
        Dictionary of tag ID to center coordinates.
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
    Apply homography transformation to the image based on fiducial coordinates.

    Args:
        fiducial_coordinates: Detected fiducial coordinates.
        image: Input image.
        row: 'A' or 'B' nozzle row.

    Returns:
        Warped image.
    """
    calibration_data = json.load(open('calibration_data.json'))

    pts_src = np.array([calibration_data[row]["homography"]], dtype=float)

    width, height = 400, 300
    pts_dst = np.array([
        [0, 0],
        [width - 1, 0],
        [0, height - 1],
        [width - 1, height - 1]
    ], dtype=float)

    print(f"Using coordinates for section {row}:")
    print(f"Source points: {pts_src}")
    print(f"Destination points: {pts_dst}")

    homography_matrix, status = cv2.findHomography(pts_src, pts_dst)
    warped_image = cv2.warpPerspective(image, homography_matrix, (width, height))

    return warped_image


def classify_nozzles(warped_image: np.ndarray, section: str = 'B', display: bool = True, ratio_thresh: float = 0.10) -> tuple[dict[str, list[bool]], float]:
    """
    Classify nozzles as clogged or not based on white ratio in warped image.

    Args:
        warped_image: Homography-warped nozzle image.
        section: 'A' or 'B'.
        display: Whether to display/save the classification image.

    Returns:
        (nozzle status dict, mean white ratio)
    """
    num_nozzles = 8
    width = warped_image.shape[1]
    height = warped_image.shape[0]

    nozzle_width = width // num_nozzles
    margin_ratio = 0.20

    white_ratios = []

    for i in range(num_nozzles):
        x_start = int(i * nozzle_width + margin_ratio * nozzle_width)
        x_end = int((i + 1) * nozzle_width - margin_ratio * nozzle_width)

        y_start = int(0.10 * height)
        y_end = int(0.90 * height)

        nozzle_region = warped_image[y_start:y_end, x_start:x_end]

        if len(nozzle_region.shape) == 3:
            nozzle_region = cv2.cvtColor(nozzle_region, cv2.COLOR_BGR2GRAY)

        white_pixels = cv2.countNonZero(nozzle_region)
        total_pixels = nozzle_region.size
        white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
        white_ratios.append(white_ratio)

    mean_ratio = np.mean(white_ratios)
    std_ratio = np.std(white_ratios)
    threshold_z = -1.75

    nozzle_status = []

    for i, ratio in enumerate(white_ratios):
        z_score = (ratio - mean_ratio) / std_ratio if std_ratio > 0 else 0
        is_clogged = z_score < threshold_z or ratio < ratio_thresh
        print(f"Nozzle {i + 1}: White ratio = {ratio:.2f}, Z-score = {z_score:.2f}, Clogged = {is_clogged}")
        nozzle_status.append(is_clogged)

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
