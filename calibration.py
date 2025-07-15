import tkinter as tk
from tkinter import ttk
from tkinter import Canvas
import json
import cv2
import numpy as np
from PIL import Image, ImageTk
from detect_clogs import threshold_video_movement

SCALING_FACTOR = 2

def calibrate_homography(filepath: str = 'output.mjpeg'):
    cycles = threshold_video_movement(filepath)
    if not cycles:
        print("No cycles detected.")
        return

    # --- Step 1: Extract fiducial coordinates and update JSON (each cycle corresponds to section A or B) ---
    frames = []
    for row, cycle in enumerate(cycles):
        frame = cycle['thresholded_image']
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        frames.append(frame)

        with open('calibration_data.json', 'r') as f:
            calibration_data = json.load(f)
        row_name = 'A' if row == 0 else 'B'
        coords = cycle['fiducial_coordinates']
        if 1 in coords:
            calibration_data[row_name]['fiducial_coordinates']["1"] = [int(coords[1][0]), int(coords[1][1])]
        if 2 in coords:
            calibration_data[row_name]['fiducial_coordinates']["2"] = [int(coords[2][0]), int(coords[2][1])]

        with open('calibration_data.json', 'w') as f:
            json.dump(calibration_data, f, indent=3)

    # --- Step 2: Initialize a calibration GUI for each frame (one window per section A or B) ---
    for row, frame in enumerate(frames):
        with open('calibration_data.json', 'r') as f:
            calibration_data = json.load(f)

        def update_homography_display():
            """Update local calibration_data when dragging red points or switching sections."""
            section = section_var.get()
            pts_src = np.array([calibration_data[section]["homography"]], dtype=float)

            for i, circle in enumerate(circles):
                x = np.mean(canvas.coords(circle)[0:4:2])
                y = np.mean(canvas.coords(circle)[1:4:2])
                x *= SCALING_FACTOR
                y *= SCALING_FACTOR
                pts_src[0][i] = [x, y]

            calibration_data[section]["homography"] = pts_src.tolist()[0]

            # Update the homography preview for both sections
            pts_src_a = np.array([calibration_data['A']["homography"]], dtype=float)
            pts_src_b = np.array([calibration_data['B']["homography"]], dtype=float)

            update_homography_image(pts_src_a, 'A')
            update_homography_image(pts_src_b, 'B')

        def update_homography_image(pts_src, row_name):
            """Apply homography transformation and draw partition boxes (with margin) for visualization."""
            width, height = 400, 300  # Target perspective image size
            pts_dst = np.array([
                [0, 0],
                [width - 1, 0],
                [0, height - 1],
                [width - 1, height - 1]
            ], dtype=float)

            homography_matrix, _ = cv2.findHomography(pts_src, pts_dst)
            warped_image = cv2.warpPerspective(frame, homography_matrix, (width, height))

            # Draw 8 nozzle regions
            num_nozzles = 8
            nozzle_width = width // num_nozzles
            margin_ratio = 0.20  # Set margin ratio (20%)

            for i in range(num_nozzles):
                x_start = i * nozzle_width
                x_end = (i + 1) * nozzle_width if (i + 1) < num_nozzles else width
                y_start = int(0.10 * height)
                y_end = int(0.90 * height)

                # Draw red outer box (full nozzle area)
                cv2.rectangle(warped_image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)

                # Calculate margin-adjusted region
                margin = int(nozzle_width * margin_ratio)
                x_start_margin = x_start + margin
                x_end_margin = x_end - margin

                # Draw white inner box (used for white ratio calculation)
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
            """Save the updated calibration_data to JSON when Save button is clicked."""
            with open('calibration_data.json', 'w') as f:
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
        img = img.resize((img.width // SCALING_FACTOR, img.height // SCALING_FACTOR))
        img_tk = ImageTk.PhotoImage(img)
        canvas_frame = ttk.Frame(mainframe)
        canvas_frame.grid(row=1, column=0, columnspan=2, pady=10)
        canvas = Canvas(canvas_frame, width=img.width, height=img.height,
                        highlightthickness=1, highlightbackground="gray")
        canvas.pack()
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

        # Initialize red points (4 homography keypoints)
        circles = []
        homography = calibration_data['A']['homography'] if row == 0 else calibration_data['B']['homography']
        for x, y in homography:
            x = x // SCALING_FACTOR
            y = y // SCALING_FACTOR
            circle = canvas.create_oval(x - 5, y - 5, x + 5, y + 5,
                                        fill='red', outline='black', width=2)
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

# Example usage
calibrate_homography('/home/jonat/Capstone/clog_detection/old_video2/segment_1_primming_B.mjpeg')
