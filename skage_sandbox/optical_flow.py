import cv2
import numpy as np

# ---------- Parameters for feature detection and optical flow ----------
# Increase maxCorners to detect more features (allowing up to 100 features)
feature_params = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

# Lucas-Kanade optical flow parameters
lk_params = dict(
    winSize=(5, 5),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Movement threshold (in pixels) to filter out noise/static points
movement_threshold = 2.0

# Colors for drawing (green for tracking, blue for velocity arrow)
track_color = (0, 255, 0)     # Green circle/line for tracked points
arrow_color = (255, 0, 0)     # Blue arrow for velocity vector

# ---------- Downsampling Parameter ----------
# process_every_n defines how many frames to skip.
# For example, process_every_n=3 means process only every 3rd frame.
process_every_n = 1

# ---------- Initialize Video Capture and Writer ----------
video_path = "skage_sandbox/data/standing.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Cannot open video file {video_path}")
    exit()

# Read the first frame and convert to grayscale
ret, old_frame = cap.read()
if not ret:
    print("Error: Unable to read the first frame.")
    cap.release()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Detect initial features in the first frame
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
if p0 is None:
    print("No features found in the first frame!")
    cap.release()
    exit()

# Setup output video writer for debugging overlay
frame_height, frame_width = old_frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_debug.avi', fourcc, 20.0, (frame_width, frame_height))

# A mask image for drawing (persisting drawn lines)
mask = np.zeros_like(old_frame)

# Lists to store projectile positions and velocities over time.
# Each element corresponds to the features detected in that frame.
projectile_positions = []  # List of lists of (x, y) tuples per frame
projectile_velocities = []  # List of lists of (dx, dy) tuples per frame

# ---------- Main Processing Loop ----------
frame_index = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if frame_index % process_every_n != 0:
        # Even though we are skipping processing,
        # update the previous frame reference to avoid too large jumps later.
        old_gray = frame_gray.copy()
        frame_index += 1
        continue

    # Calculate optical flow from previous frame to current frame
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    if p1 is None or st is None:
        # If tracking fails, try to re-detect features in the previous frame
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        continue

    # Select only the successfully tracked points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    frame_positions = []  # positions for this frame
    frame_velocities = []  # velocities for this frame

    # Process each tracked point
    for new_pt, old_pt in zip(good_new, good_old):
        a, b = new_pt.ravel()
        c, d = old_pt.ravel()

        # Compute the displacement vector (velocity)
        dx = a - c
        dy = b - d
        velocity_magnitude = np.sqrt(dx**2 + dy**2)

        # If the movement is significant enough, assume it belongs to a moving projectile
        if velocity_magnitude > movement_threshold:
            frame_positions.append((a, b))
            frame_velocities.append((dx, dy))

            # Draw the trajectory line on the mask (from previous to current point)
            mask = cv2.line(mask, (int(c), int(d)), (int(a), int(b)), track_color, 2)
            # Draw a circle at the current position
            frame = cv2.circle(frame, (int(a), int(b)), 5, track_color, -1)
            # Draw an arrow indicating the velocity vector (scaled for visualization)
            arrow_scale = 5  # Adjust for clarity
            end_point = (int(a + arrow_scale * dx), int(b + arrow_scale * dy))
            frame = cv2.arrowedLine(frame, (int(a), int(b)), end_point, arrow_color, 2)

    # Save positions and velocities for this frame
    projectile_positions.append(frame_positions)
    projectile_velocities.append(frame_velocities)

    # If the number of tracked features falls below a threshold (e.g., 20), re-detect features.
    if len(good_new) < 20:
        # You can use a mask here if needed to avoid detecting already-tracked points
        new_features = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        if new_features is not None:
            # Combine the new features with the current good_new features
            p0 = np.concatenate((good_new.reshape(-1, 1, 2), new_features), axis=0)
        else:
            p0 = good_new.reshape(-1, 1, 2)
    else:
        p0 = good_new.reshape(-1, 1, 2)

    # Overlay the tracking mask on the frame
    img_overlay = cv2.add(frame, mask)

    # Write the debug frame to the output video
    out.write(img_overlay)

    # Optional: Display the frame (press 'q' to quit early)
    cv2.imshow('Optical Flow Tracking', img_overlay)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # Update for the next iteration
    old_gray = frame_gray.copy()
    frame_index += 1

# ---------- Cleanup ----------
cap.release()
out.release()
cv2.destroyAllWindows()

# For debugging purposes, print out the collected positions and velocities (per frame)
print("Collected Projectile Positions (per frame):", projectile_positions)
print("Collected Projectile Velocities (per frame):", projectile_velocities)