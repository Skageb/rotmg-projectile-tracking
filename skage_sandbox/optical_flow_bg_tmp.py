import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def gaussian_downsize(frame):
    # Apply Gaussian blur and downsize the image for efficiency and denoising
    blurred_frame = cv.GaussianBlur(frame, (19, 19), 1.2)
    frame = cv.resize(blurred_frame,
                      (frame.shape[1] // 2, frame.shape[0] // 2),
                      interpolation=cv.INTER_AREA)
    return frame



CREATE_DEMO_MP4 = True
RESULT_FILE = 'optical_flow/OF_3rd_frame.mp4'
RESULT_ORIGINAL = 'optical_flow/OF_3rd_frame_original.mp4'


N_FRAMES_SKIP = 0
INSPECT_FRAMES = True
MAGNITUDE_HEATMAP = True

cap = cv.VideoCapture('data/movement.mp4')
ret, first_frame = cap.read()


if not ret:
    print("Error reading video.")
    cap.release()
    exit()

prev_raw = first_frame


first_frame = gaussian_downsize(first_frame)
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

frame_height, frame_width = first_frame.shape[:2]

# Create an HSV mask for *visualizing* flow (Hue, Saturation=255, Value)
mask = np.zeros_like(first_frame)
mask[..., 1] = 255  # Full saturation

fourcc = cv.VideoWriter_fourcc(*'mp4v')
out_optical = cv.VideoWriter(f'results/{RESULT_FILE}', fourcc, 5.0, (frame_width, frame_height))
out_original = cv.VideoWriter(f'results/{RESULT_ORIGINAL}', fourcc, 5.0, (frame_width, frame_height))

diff_log = []


if MAGNITUDE_HEATMAP:
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    # Initialize imshow with some dummy data so we can set the size correctly
    dummy_image = np.zeros((frame_height, frame_width), dtype=np.float32)
    im = ax.imshow(dummy_image, vmin=0, vmax=20, cmap='hot')  # Force scale 0..20
    cbar = plt.colorbar(im, ax=ax, label='Flow magnitude')
    ax.set_title('Optical Flow Magnitude Heatmap')
    plt.show(block=False)  # Non-blocking show


while cap.isOpened():
    for n in range(N_FRAMES_SKIP+1):  #Skip N frames
        ret, frame = cap.read()


    if not ret:
        break

    raw_frame = frame

    diff = np.sum(cv.absdiff(raw_frame, prev_raw))
    #print('Diff current and previous frame', np.sum(diff))

    diff_log.append(diff)
    if len(diff_log) > 10:
        diff_log.pop(0)
    if diff < np.average(diff_log)/8:
        print('skipped_frame')
        continue

    print(diff)
    
    frame = gaussian_downsize(frame)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #--- 1) Compute Flow ---
    flow = cv.calcOpticalFlowFarneback(
        prev_gray, gray,
        None,       # no initial flow
        0.8, 3, 15, 3, 5, 1.2, 0
    )
    # flow[..., 0] = x-flow, flow[..., 1] = y-flow

    #--- 2) Convert Flow to Magnitude and Angle (in Radians) ---
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)

    if MAGNITUDE_HEATMAP:
        im.set_data(magnitude)   # Just update data
        ax.set_title('Optical Flow Magnitude')  # Optionally update title
        plt.pause(0.001)   


    #--- 3) Find the Dominant (Background) Flow Vector ---
    #    3a) Convert angle to "HSV hue space" for quick binning [0..180] range
    #        (OpenCV HSV expects hue in [0..180], so we scale from [0..2pi]).
    #        The mask is just used for counting the “most frequent direction”.
    hue_for_binning = angle * 180 / np.pi / 2  # from [0..2π] to [0..180]

    #    3b) Pick the most frequent direction. We'll do a coarse bin by rounding.
    #        If you want finer control, you could pick a different bin size.
    hue_rounded = np.round(hue_for_binning).astype(np.uint8)
    directions, dir_counts = np.unique(hue_rounded, return_counts=True)
    dominant_hue = directions[np.argmax(dir_counts)]  # Hue in [0..180] scale

    #    3c) From that direction, find the median or mode magnitude among those pixels.
    #        We'll do a quick approach: gather all pixel magnitudes with that hue_rounded.
    idx = (hue_rounded == dominant_hue)
    mag_values_in_dom_dir = magnitude[idx]

    # If everything truly is background, you might have all or most pixels in the same direction.
    # We can pick the median magnitude for stability (less sensitive to outliers).
    if len(mag_values_in_dom_dir) == 0:
        # Fallback if something strange happens
        bg_magnitude = 0.0
    else:
        bg_magnitude = np.median(mag_values_in_dom_dir)

    # Convert the dominant hue back to radians:
    # dominant_hue in [0..180], so we convert back: hue_degrees = dominant_hue*2
    bg_angle_degs = dominant_hue * 2.0
    bg_angle_rad = (bg_angle_degs * np.pi) / 180.0

    #--- 4) Compute the Background Flow Vector (X,Y) ---
    bg_flow_x = bg_magnitude * np.cos(bg_angle_rad)
    bg_flow_y = bg_magnitude * np.sin(bg_angle_rad)

    #--- 5) Subtract the Background from Each Pixel’s Flow ---
    residual_flow_x = flow[..., 0] - bg_flow_x
    residual_flow_y = flow[..., 1] - bg_flow_y

    #--- 6) Compute Residual Magnitude/Angle for Visualization ---
    res_magnitude, res_angle = cv.cartToPolar(residual_flow_x, residual_flow_y, angleInDegrees=False)

    #--- 7) Update the HSV Mask for Display (Residual) ---
    # Hue in [0..180] range => res_angle in [0..2π]
    mask[..., 0] = (res_angle * 180 / np.pi / 2).astype(np.uint8)
    # Full saturation
    mask[..., 1] = 255
    # Value channel: normalize residual magnitude for display
    mask[..., 2] = cv.normalize(res_magnitude, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    # Convert HSV to BGR or RGB for display
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    
    cv.imshow('Input', frame)
    cv.imshow('Residual Flow', rgb)

    #--- 8) Update for Next Loop ---
    prev_gray = gray.copy()

    prev_raw = raw_frame
    prev_gray = gray

    out_optical.write(rgb)
    out_original.write(gray)

    
    if INSPECT_FRAMES:
        print("Press Enter to see next frame (or Esc to exit)...")
        while True:
            key = cv.waitKey(0) & 0xFF  # Wait indefinitely until a key is pressed
            # If Enter is pressed, break out of the waiting loop; if Esc is pressed, exit the main loop.
            if key == 13:  # Enter key
                break
            elif key == 27:  # Escape key
                cap.release()
                out_optical.release()
                out_original.release()
                cv.destroyAllWindows()
                exit()
    
    else:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break   

cap.release()
cv.destroyAllWindows()
