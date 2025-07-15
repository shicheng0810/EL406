import tkinter as tk
from tkinter import ttk
from tkinter import Canvas
import json
import cv2
import numpy as np
from PIL import Image, ImageTk
from detect_clogs import *

SCALING_FACTOR = 2
ENLARGE_FACTOR = 2

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
                    adjusted_coordinates[key] = type('obj', (object,), {'x': coord[0] - x, 'y': coord[1] - y})
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
        # Avoid 0 error
        if frame_count > 0:
            accumulated_mask /= frame_count

        # Convert accumulated mask to 8-bit image
        # Avoid Nan
        accumulated_mask = np.nan_to_num(accumulated_mask)
        accum_mask_uint8 = cv2.normalize(accumulated_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply thresholding
        _, thresholded_image = cv2.threshold(accum_mask_uint8, 0, 255, cv2.THRESH_BINARY)

        cycle['thresholded_image'] = thresholded_image
        cycle['fiducial_coordinates'] = fiducial_coordinates
        # 保存ROI信息以供后续使用
        if roi is not None:
            cycle['roi'] = roi

    # Save ROI
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
    
    # Create ROI window
    roi_window = tk.Tk()
    roi_window.title("Choose Region of Interest")
    
    # Save ROI
    roi_data = {'start_x': 0, 'start_y': 0, 'end_x': 0, 'end_y': 0, 'confirmed': False}
    
    # Adjust picture 
    img_pil = Image.fromarray(frame)
    img_width = img_pil.width // SCALING_FACTOR
    img_height = img_pil.height // SCALING_FACTOR
    img_pil = img_pil.resize((img_width, img_height))
    img_tk = ImageTk.PhotoImage(img_pil)
    
    # Create Canvas
    canvas = tk.Canvas(roi_window, width=img_width, height=img_height)
    canvas.pack(padx=10, pady=10)
    
    # Save img
    canvas.img_tk = img_tk
    
    # Show img
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

def calibrate_homography(filepath: str = 'output.mjpeg'):
    cycles = threshold_video_movement(filepath)
    if not cycles:
        print("No cycles detected.")
        return

    # --- Step 1: Extract fiducial coordinates and update JSON (each cycle corresponds to section A or B) ---
    frames = []
    for row, cycle in enumerate(cycles):
        frame = cycle['thresholded_image']
        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

        # Read and update fiducial coordinates in the JSON file
        with open('calibration_data_plate.json', 'r') as f:
            calibration_data = json.load(f)
        row_name = 'A' if row == 0 else 'B'
        coords = cycle['fiducial_coordinates']
        calibration_data[row_name]['roi'] = cycle.get('roi')
        if 1 in coords:
            calibration_data[row_name]['fiducial_coordinates']["1"] = [int(coords[1].x), int(coords[1].y)]
        if 2 in coords:
            calibration_data[row_name]['fiducial_coordinates']["2"] = [int(coords[2].x), int(coords[2].y)]

        # Save back to JSON file
        with open('calibration_data_plate.json', 'w') as f:
            json.dump(calibration_data, f, indent=3)

    # --- Step 2: Initialize a simple calibration GUI for each frame (assuming separate popups for A & B) ---
    for row, frame in enumerate(frames):
        # Load calibration data once (no repeated reading)
        with open('calibration_data_plate.json', 'r') as f:
            calibration_data = json.load(f)

        def update_homography_display():
            """Update local calibration_data when dragging red points or switching sections."""
            section = section_var.get()

            # Use the already loaded calibration_data instead of reloading from file
            pts_src = np.array([calibration_data[section]["homography"]], dtype=float)

            # Extract updated red point positions (convert back using SCALING_FACTOR)
            for i, circle in enumerate(circles):
                x = np.mean(canvas.coords(circle)[0:4:2])
                y = np.mean(canvas.coords(circle)[1:4:2])
                x /= ENLARGE_FACTOR
                y /= ENLARGE_FACTOR
                pts_src[0][i] = [x, y]

            # Update local calibration_data
            calibration_data[section]["homography"] = pts_src.tolist()[0]

            # Update the homography preview for sections A & B
            # pts_src_a = np.array([calibration_data['A']["homography"]], dtype=float)
            # pts_src_b = np.array([calibration_data['B']["homography"]], dtype=float)

            pts_src_a = np.array([calibration_data['A']["homography"]], dtype=float)
            update_homography_image(pts_src_a, 'A', cycles[0].get('roi'))

            if len(cycles) > 1:
                pts_src_b = np.array([calibration_data['B']["homography"]], dtype=float)
                update_homography_image(pts_src_b, 'B', cycles[1].get('roi'))

        def update_homography_image(pts_src, row_name, roi):
            """Perform homography transformation and draw partition boxes (with margin) for visualization."""
            # Get ROI size from the corresponding cycle
            if roi is not None:
                _, _, w, h = roi
                width, height = w, h
            else:
                width, height = 400, 300  # fallback default

            # width, height = 400, 300  #Image size
            pts_dst = np.array([
                [0, 0],
                [width - 1, 0],
                [0, height - 1],
                [width - 1, height - 1]
            ], dtype=float)

            homography_matrix, _ = cv2.findHomography(pts_src, pts_dst)
            warped_image = cv2.warpPerspective(frame, homography_matrix, (width, height))

            # Draw 8-nozzle area
            num_nozzles = 8
            nozzle_width = width // num_nozzles
            margin_ratio = 0.20  # Set margin（10%）

            for i in range(num_nozzles):
                x_start = i * nozzle_width
                x_end = (i + 1) * nozzle_width if (i + 1) < num_nozzles else width
                y_start = int(0.10 * height)
                y_end = int(0.90 * height)

                # Draw red frame
                cv2.rectangle(warped_image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)

                # Calculate margin area
                margin = int(nozzle_width * margin_ratio)
                x_start_margin = x_start + margin
                x_end_margin = x_end - margin

                # Draw white frame
                cv2.rectangle(warped_image, (x_start_margin, y_start), (x_end_margin, y_end), (255, 255, 255), 2)

        
            img = Image.fromarray(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(img)
            
            if row_name == 'A':
                homography_label_A.config(image=img_tk)
                homography_label_A.image = img_tk
            else:
                homography_label_B.config(image=img_tk)
                homography_label_B.image = img_tk

        def on_circle_drag(event):
            """Update coordinates and refresh preview when user drags red points."""
            item = canvas.find_withtag("current")
            canvas.coords(item, event.x - 5, event.y - 5, event.x + 5, event.y + 5)
            update_homography_display()

        def save_calibration_data():
            """Save the updated calibration_data to JSON when the Save button is clicked."""
            with open('calibration_data_plate.json', 'w') as f:
                json.dump(calibration_data, f, indent=3)

            section = section_var.get()
            status_var.set(f"Calibration data for section {section} saved.")

        # ---------------------- Create the GUI Window ----------------------
        root = tk.Tk()
        root.title("Homography Calibration")

        style = ttk.Style()
        style.theme_use('clam')

        mainframe = ttk.Frame(root, padding="10 10 10 10")
        mainframe.grid(row=0, column=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # Dropdown to select section A or B
        section_var = tk.StringVar()
        section_var.set('A' if row == 0 else 'B')
        section_label = ttk.Label(mainframe, text="Select Section:")
        section_label.grid(row=0, column=0, sticky=tk.W)
        section_dropdown = ttk.Combobox(mainframe, textvariable=section_var, values=['A', 'B'], state='readonly')
        section_dropdown.grid(row=0, column=1, sticky=tk.W)
        section_dropdown.bind('<<ComboboxSelected>>', lambda e: update_homography_display())

        # Display the frame image with red points on a Canvas
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = img.resize((img.width *2, img.height *2))
        img_tk = ImageTk.PhotoImage(img)
        canvas_frame = ttk.Frame(mainframe)
        canvas_frame.grid(row=1, column=0, columnspan=2, pady=10)
        canvas = Canvas(canvas_frame, width=img.width, height=img.height,
                        highlightthickness=1, highlightbackground="gray")
        canvas.pack()
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

        # Initialize red points (4 keypoints)
        circles = []
        homography = calibration_data['A']['homography'] if row == 0 else calibration_data['B']['homography']
        for x, y in homography:
            # Scale to fit Canvas
            x = x * ENLARGE_FACTOR
            y = y * ENLARGE_FACTOR
            circle = canvas.create_oval(x - 5, y - 5, x + 5, y + 5,
                                        fill='red', outline='black', width=2)
            # Bind drag event
            canvas.tag_bind(circle, '<B1-Motion>', on_circle_drag)
            circles.append(circle)

        # Save button
        button_frame = ttk.Frame(mainframe)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        save_button = ttk.Button(button_frame, text="Save", command=save_calibration_data)
        save_button.pack()

        # Homography preview labels for sections A & B
        homography_frame = ttk.Frame(mainframe, padding="10 10 10 10")
        homography_frame.grid(row=3, column=0, columnspan=2)

        label_A = ttk.Label(homography_frame, text="Section A", font=("Helvetica", 12, 'bold'))
        label_A.grid(row=0, column=0, padx=10)
        homography_label_A = ttk.Label(homography_frame)
        homography_label_A.grid(row=1, column=0, padx=10)

        label_B = ttk.Label(homography_frame, text="Section B", font=("Helvetica", 12, 'bold'))
        label_B.grid(row=0, column=1, padx=10)
        homography_label_B = ttk.Label(homography_frame)
        homography_label_B.grid(row=1, column=1, padx=10)

        # Status bar
        status_var = tk.StringVar()
        status_var.set("Drag the red points to adjust the homography. Select section A or B.")
        status_bar = ttk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))

        update_homography_display()

        root.mainloop()

calibrate_homography('/home/jonat/Capstone/clog_detection/tmp/segment_2_dispensing_B.mjpeg')  
