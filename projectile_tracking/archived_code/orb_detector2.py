import cv2
import numpy as np
import math

# ============================
# Parameters and Setup
# ============================
CREATE_DEMO_MP4 = True

# Downsampling parameter: process only every nth frame
process_every_n = 3

# Parameters for SimpleBlobDetector tuned for circular orbs.
# Note: We have removed the green color filter, so we disable color filtering.
blob_params = cv2.SimpleBlobDetector_Params()
blob_params.filterByColor = False  # Disable color filtering

blob_params.filterByArea = True
blob_params.minArea = 50       # Adjust based on expected orb size
blob_params.maxArea = 5000     # Adjust as needed

blob_params.filterByCircularity = True
blob_params.minCircularity = 0.7

blob_params.filterByInertia = True
blob_params.minInertiaRatio = 0.5

# Create the blob detector with the specified parameters.
detector = cv2.SimpleBlobDetector_create(blob_params)

# Maximum allowed distance (in pixels) when matching blob positions between frames.
max_match_distance = 50

# Scale factor for drawing velocity arrows.
arrow_scale = 5

# ============================
# Video I/O Setup
# ============================
video_path = "data/standing.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: cannot open video file")
    exit()

# Read the first frame (for dimension purposes)
ret, frame = cap.read()
if not ret:
    print("Error: cannot read first frame")
    cap.release()
    exit()

frame_height, frame_width = frame.shape[:2]
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output_debug_blob_nocolor.avi', fourcc, 20.0, (frame_width, frame_height))


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('results/demo.mp4', fourcc, 5.0, (frame_width, frame_height))

# ============================
# Storage for Debugging
# ============================
projectile_positions = []  # List of lists of blob positions per processed frame.
projectile_velocities = [] # List of lists of velocity vectors per processed frame.

# For matching blob positions between frames.
prev_positions = None

frame_index = 0

# ============================
# Main Processing Loop
# ============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Downsample: only process every process_every_n frame.
    if frame_index % process_every_n != 0:
        frame_index += 1
        continue

    # Convert frame to grayscale (no color filter is applied now).
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Optionally, you can apply a threshold here to enhance contrast:
    # ret_val, gray = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Detect blobs on the grayscale image.
    keypoints = detector.detect(gray)

    # Extract current blob centers.
    current_positions = [kp.pt for kp in keypoints]

    # Compute velocity vectors by matching current positions to previous ones.
    current_velocities = []
    if prev_positions is None:
        # First processed frame; no velocity information.
        current_velocities = [(0, 0)] * len(current_positions)
    else:
        # For each current blob, find the nearest blob in the previous frame.
        used_prev = [False] * len(prev_positions)
        for cur in current_positions:
            best_dist = float('inf')
            best_index = -1
            for idx, prev in enumerate(prev_positions):
                if not used_prev[idx]:
                    dist = math.hypot(cur[0] - prev[0], cur[1] - prev[1])
                    if dist < best_dist:
                        best_dist = dist
                        best_index = idx
            if best_index != -1 and best_dist < max_match_distance:
                dx = cur[0] - prev_positions[best_index][0]
                dy = cur[1] - prev_positions[best_index][1]
                current_velocities.append((dx, dy))
                used_prev[best_index] = True
            else:
                # No close match was found; set velocity to zero.
                current_velocities.append((0, 0))

    # Save positions and velocities for debugging purposes.
    projectile_positions.append(current_positions)
    projectile_velocities.append(current_velocities)

    # Draw the detected blobs and velocity arrows on the frame.
    for pos, vel in zip(current_positions, current_velocities):
        x, y = int(pos[0]), int(pos[1])
        cv2.circle(frame, (x, y), 7, (0, 255, 0), -1)
        # Only draw an arrow if a non-zero velocity is detected.
        if vel != (0, 0):
            end_point = (int(x + arrow_scale * vel[0]), int(y + arrow_scale * vel[1]))
            cv2.arrowedLine(frame, (x, y), end_point, (255, 0, 0), 2)

    # Write the processed frame with overlays to the output video.
    out.write(frame)
    cv2.imshow('Blob Detection and Tracking', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # Update previous positions for the next iteration.
    prev_positions = current_positions
    frame_index += 1

# ============================
# Cleanup
# ============================
cap.release()
out.release()
cv2.destroyAllWindows()

# Debug: Print out the collected positions and velocities.
print("Collected Projectile Positions (per processed frame):", projectile_positions)
print("Collected Projectile Velocities (per processed frame):", projectile_velocities)

print(len(projectile_positions))
print(len(projectile_velocities))