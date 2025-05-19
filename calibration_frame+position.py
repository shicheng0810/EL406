import numpy as np
from sklearn.metrics import f1_score
from typing import List, Dict, Tuple
import tkinter as tk
from tkinter import messagebox
from detect_clogs_realtime import threshold_video_movement_realtime_with_config
import json
import cv2
from tkinter import ttk, Canvas
from PIL import Image, ImageTk
from detect_clogs import threshold_video_movement


def calibrate_threshold(
    video_path: str,
    ground_truth: Dict[str, List[bool]],
    test_ratios: List[float],
    test_initial_frames: List[int],
    detection_fn,
    display: bool = False
) -> Tuple[float, int]:
    best_score = -1
    best_params = (test_ratios[0], test_initial_frames[0])

    for ratio_thresh in test_ratios:
        for initial_frame in test_initial_frames:
            # print(f"Testing ratio={ratio_thresh}, initial_frame={initial_frame}")

            prediction = detection_fn(
                video_path=video_path,
                ratio_thresh=ratio_thresh,
                initial_frame_num=initial_frame,
                display=display
            )

            if prediction is None:
                continue

            score = evaluate_predictions(prediction, ground_truth)
            # print(f"F1 score: {score:.3f}")

            if score > best_score:
                best_score = score
                best_params = (ratio_thresh, initial_frame)

    print(f"\nBest config: ratio={best_params[0]}, start_frame={best_params[1]}, score={best_score:.3f}")
    return best_params


def evaluate_predictions(predicted: Dict[str, List[bool]], ground_truth: Dict[str, List[bool]]) -> float:
    """
    Evaluate the prediction performance using F1 score.
    Special case handling: if the ground truth has no clogs (all False),
    assign a score of 1.0 if prediction also has no clogs, else 0.0.

    Args:
        predicted: Dictionary of predicted clog status by section (e.g., {'A': [False, True, ...]})
        ground_truth: Dictionary of true clog status by section

    Returns:
        Mean F1 score across sections
    """
    scores = []
    for section in ground_truth:
        if section not in predicted:
            continue
        gt = ground_truth[section]
        pred = predicted[section]
        if len(gt) != len(pred):
            continue
        if sum(gt) == 0:
            # No true clogs: if model predicts all clear, give full score; else zero
            score = 1.0 if sum(pred) == 0 else 0.0
        else:
            score = f1_score(gt, pred)
        scores.append(score)
    return np.mean(scores) if scores else 0.0


def launch_threshold_gui(video_path: str):
    def run_calibration():
        section = section_var.get()
        gt = [bool(var.get()) for var in (nozzle_vars_A if section == "A" else nozzle_vars_B)]
        ground_truth = {section: gt}
        print("Ground Truth from GUI:", ground_truth)

        best_thresh, best_frame = calibrate_threshold(
            video_path,
            ground_truth,
            test_ratios=[0.03, 0.05, 0.07, 0.08, 0.1, 0.12],  
            test_initial_frames=[30, 50, 70],
            detection_fn=threshold_video_movement_realtime_with_config
        )

        config = {
            "best_threshold": best_thresh,
            "start_frame": best_frame
        }
        with open("calibration_threshold.json", "w") as f:
            json.dump(config, f, indent=4)

        result_var.set(f"Best Ratio Threshold: {best_thresh}, Start Frame: {best_frame}")
        print("Saved to calibration_threshold.json")

        print("Running nozzle classification with best parameters...")
        result = threshold_video_movement_realtime_with_config(
            video_path=video_path,
            ratio_thresh=best_thresh,
            initial_frame_num=best_frame,
            display=True
        )
        print("Final classification result:", result)

    def update_checkboxes(*args):
        section = section_var.get()
        current_vars = nozzle_vars_A if section == "A" else nozzle_vars_B
        for i in range(8):
            checkboxes[i].config(variable=current_vars[i])

    root = tk.Tk()
    root.title("Threshold Calibration")

    tk.Label(root, text="Select Section:").pack()
    section_var = tk.StringVar(value="A")
    section_menu = tk.OptionMenu(root, section_var, "A", "B")
    section_menu.pack()

    tk.Label(root, text="Select Clogged Nozzles (Ground Truth):").pack()
    nozzle_frame = tk.Frame(root)
    nozzle_frame.pack()

    nozzle_vars_A = [tk.IntVar() for _ in range(8)]
    nozzle_vars_B = [tk.IntVar() for _ in range(8)]
    checkboxes = []

    for i in range(8):
        cb = tk.Checkbutton(nozzle_frame, text=f"Nozzle {i+1}", variable=nozzle_vars_A[i])  # 默认显示A
        cb.grid(row=i // 4, column=i % 4)
        checkboxes.append(cb)

    section_var.trace_add("write", update_checkboxes)

    tk.Button(root, text="Run Calibration", command=run_calibration).pack(pady=10)

    result_var = tk.StringVar()
    tk.Label(root, textvariable=result_var, fg="green").pack()

    root.mainloop()



video_file = "/home/jonat/Capstone/clog_detection/tmp/segment_1_primming.mjpeg"
# calibrate_homography(video_file)
launch_threshold_gui(video_file)