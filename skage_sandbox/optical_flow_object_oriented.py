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


class ProjectileTracker:
    def __init__(self):
        # Configuration flags
        self.CREATE_DEMO_MP4 = True
        self.INSPECT_FRAMES = True
        self.MAGNITUDE_HEATMAP = True

        self.N_FRAMES_SKIP = 0

        # File output settings
        self.result_file = 'optical_flow/OF_3rd_frame.mp4'
        self.original_video_file = 'optical_flow/OF_3rd_frame_original.mp4'

        # Will be set up later
        self.cap = None
        self.out_optical = None
        self.out_original = None

        # Internal tracking variables
        self.frame_width = None
        self.frame_height = None
        self.prev_gray = None
        self.prev_raw = None
        self.diff_log = []

        # For heatmap visualization
        self.fig = None
        self.ax = None
        self.im = None

    def __init_video__(self, video_path):
        """Initialize video capture, read first frame, set up variables."""
        self.cap = cv.VideoCapture(video_path)
        ret, first_frame = self.cap.read()
        if not ret:
            print("Error reading video.")
            if self.cap is not None:
                self.cap.release()
            exit()

        # Keep a copy of the raw frame for difference checks
        self.prev_raw = first_frame

        # Downsize and convert to grayscale
        first_frame = self.__gaussian_downsize__(first_frame)
        self.prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

        # Dimensions
        self.frame_height, self.frame_width = first_frame.shape[:2]

        # Prepare optical flow mask for visualization
        # (Hue, Saturation=255, Value to be set)
        self.mask = np.zeros_like(first_frame)
        self.mask[..., 1] = 255  # Full saturation

        # Prepare video writers if needed
        if self.CREATE_DEMO_MP4:
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            self.out_optical = cv.VideoWriter(self.result_file, fourcc, 5.0,
                                              (self.frame_width, self.frame_height))
            self.out_original = cv.VideoWriter(self.original_video_file, fourcc, 5.0,
                                               (self.frame_width, self.frame_height))

        # Set up heatmap if requested
        if self.MAGNITUDE_HEATMAP:
            plt.ion()
            self.fig, self.ax = plt.subplots()
            dummy_image = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)
            self.im = self.ax.imshow(dummy_image, vmin=0, vmax=20, cmap='hot')
            plt.colorbar(self.im, ax=self.ax, label='Flow magnitude')
            self.ax.set_title('Optical Flow Magnitude Heatmap')
            plt.show(block=False)

    def __gaussian_downsize__(self, frame):
        """Utility to blur and downsize a frame for efficiency."""
        blurred_frame = cv.GaussianBlur(frame, (19, 19), 1.2)
        frame = cv.resize(blurred_frame,
                          (frame.shape[1] // 2, frame.shape[0] // 2),
                          interpolation=cv.INTER_AREA)
        return frame
    

    def __compute_optical_flow__(self, prev_gray, gray, **OpticalFlow_Args):
        """Compute Farneback optical flow."""
        flow = False

        required_args = {
            "pyr_scale", "levels", "winsize", "iterations", 
            "poly_n", "poly_sigma", "flags"
        }

        if OpticalFlow_Args and (set(OpticalFlow_Args.keys()) & required_args) == set(OpticalFlow_Args.keys()):
            flow = cv.calcOpticalFlowFarneback(
                prev_gray, gray,
                flow=None,          # no initial flow
                **OpticalFlow_Args
            )
        if not flow:
            flow = cv.calcOpticalFlowFarneback(
                prev_gray, gray,
                flow=None,          # no initial flow
                pyr_scale=0.8, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
        return flow
    

    def __pixel_clustering__(
        self, residual_flow_x, residual_flow_y, frame_for_drawing,
        mag_threshold=2.0, size_threshold=10, num_bins=36
    ):
        """
        Returns a list of tuples: (bin_id, (min_x, min_y, max_x, max_y)).
        We can merge only those that share the same bin_id if desired.
        """
        res_magnitude, res_angle = cv.cartToPolar(residual_flow_x, residual_flow_y, angleInDegrees=False)
        magnitude_mask = (res_magnitude > mag_threshold)

        angle_degs = np.degrees(res_angle)
        angle_bin = (angle_degs // (360 / num_bins)).astype(np.uint8)

        bounding_boxes = []
        for b in range(num_bins):
            mask_b = ((angle_bin == b) & magnitude_mask).astype(np.uint8)
            if np.count_nonzero(mask_b) == 0:
                continue

            num_labels, labels_img = cv.connectedComponents(mask_b, connectivity=8)
            for label_id in range(1, num_labels):
                coords = np.where(labels_img == label_id)
                size = len(coords[0])
                if size < size_threshold:
                    continue

                min_y, max_y = coords[0].min(), coords[0].max()
                min_x, max_x = coords[1].min(), coords[1].max()

                # Draw the bounding box on the "frame_for_drawing" if you wish
                # cv.rectangle(frame_for_drawing, (min_x, min_y), (max_x, max_y), (0,255,0), 2)

                bounding_boxes.append((b, (min_x, min_y, max_x, max_y)))

        return bounding_boxes
    
    def __boxes_overlap__(self, boxA, boxB, margin=5):
        """
        boxA, boxB: (min_x, min_y, max_x, max_y)
        Returns True if the boxes overlap (non-zero intersection).
        """
        (Ax1, Ay1, Ax2, Ay2) = boxA
        (Bx1, By1, Bx2, By2) = boxB

        # Check for no-overlap in X or Y
        if Ax2+margin < Bx1 or Bx2+margin < Ax1:
            return False
        if Ay2+margin < By1 or By2+margin < Ay1:
            return False
        return True

    def __merge_boxes__(self, boxA, boxB):
        """
        Merge two boxes into one bounding box that covers both (the union).
        """
        (Ax1, Ay1, Ax2, Ay2) = boxA
        (Bx1, By1, Bx2, By2) = boxB
        return (
            min(Ax1, Bx1),
            min(Ay1, By1),
            max(Ax2, Bx2),
            max(Ay2, By2)
        )

    def __filter_and_merge_bounding_boxes__(self, bounding_boxes, min_l, max_l):
        """
        bounding_boxes: list of (bin_id, (min_x, min_y, max_x, max_y))
        max_area: bounding boxes with area > max_area are dropped
        
        Returns a new list of merged, filtered bounding boxes in the same format:
        [ (bin_id, (min_x, min_y, max_x, max_y)), ... ]
        """
        # 1) Filter out boxes that are too big up-front (optional to do after merging instead)
        filtered_boxes = []
        for (b_id, (x1, y1, x2, y2)) in bounding_boxes:
            w, h = (x2 - x1), (y2 - y1)
            if min_l <= w <= max_l and min_l <= h <= max_l:
                filtered_boxes.append((b_id, (x1, y1, x2, y2)))

        # 2) We'll do naive iterative merging of overlapping boxes that share the same bin_id
        merged = True
        while merged:
            merged = False
            result = []
            while filtered_boxes:
                curr_bin, curr_box = filtered_boxes.pop()
                # Try to find a box in 'result' that overlaps with this one
                merged_index = None
                for i, (r_bin, r_box) in enumerate(result):
                    if r_bin == curr_bin and self.__boxes_overlap__(curr_box, r_box):
                        # Merge them
                        new_box = self.__merge_boxes__(curr_box, r_box)
                        # replace the box in result
                        result[i] = (r_bin, new_box)
                        merged = True
                        merged_index = i
                        break

                if merged_index is None:
                    # No overlap found => keep it
                    result.append((curr_bin, curr_box))
            filtered_boxes = result

        return filtered_boxes
    

    def __subtract_background__(self, flow):
        """Find dominant background flow direction & magnitude, subtract it from flow."""
        # 1) Convert flow to magnitude/angle
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)

        # 2) Convert angle to [0..180] range for quick binning (like HSV hue)
        hue_for_binning = angle * 180 / np.pi / 2
        hue_rounded = np.round(hue_for_binning).astype(np.uint8)

        # 3) Find the most frequent direction (dominant hue)
        directions, dir_counts = np.unique(hue_rounded, return_counts=True)
        dominant_hue = directions[np.argmax(dir_counts)]

        # 4) Among those pixels in that hue, find the median magnitude
        idx = (hue_rounded == dominant_hue)
        mag_values_in_dom_dir = magnitude[idx]
        if len(mag_values_in_dom_dir) == 0:
            bg_magnitude = 0.0
        else:
            bg_magnitude = np.median(mag_values_in_dom_dir)

        # 5) Convert hue back to radians
        bg_angle_degs = dominant_hue * 2.0
        bg_angle_rad = (bg_angle_degs * np.pi) / 180.0

        bg_flow_x = bg_magnitude * np.cos(bg_angle_rad)
        bg_flow_y = bg_magnitude * np.sin(bg_angle_rad)

        # 6) Subtract background from original flow
        residual_flow_x = flow[..., 0] - bg_flow_x
        residual_flow_y = flow[..., 1] - bg_flow_y

        return residual_flow_x, residual_flow_y
    

    def __update_heatmap__(self, magnitude):
        """Update the magnitude heatmap in real time, if enabled."""
        if self.MAGNITUDE_HEATMAP and self.im is not None:
            self.im.set_data(magnitude)
            self.ax.set_title('Optical Flow Magnitude Heatmap')
            plt.pause(0.001)

    
    def __track_projectiles__(self):
        """Main loop to read frames, compute flow, subtract background, detect projectiles, etc."""
        while self.cap.isOpened():
            # Skip N frames if needed
            for _ in range(self.N_FRAMES_SKIP + 1):
                ret, frame = self.cap.read()

            if not ret:
                break

            # Check difference from previous frame (to maybe skip near-identical frames)
            diff = np.sum(cv.absdiff(frame, self.prev_raw))
            self.diff_log.append(diff)
            if len(self.diff_log) > 10:
                self.diff_log.pop(0)

            if diff < np.average(self.diff_log) / 8:
                print('skipped_frame')
                continue

            raw_frame = frame

            # Downsize and convert current frame
            frame = self.__gaussian_downsize__(frame)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Compute optical flow
            flow = self.__compute_optical_flow__(self.prev_gray, gray)

            # Compute & visualize magnitude if needed
            magnitude, _ = cv.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
            self.__update_heatmap__(magnitude)

            # Subtract background
            residual_flow_x, residual_flow_y = self.__subtract_background__(flow)

            

            # Build residual HSV for display
            res_magnitude, res_angle = cv.cartToPolar(residual_flow_x, residual_flow_y, angleInDegrees=False)
            self.mask[..., 0] = (res_angle * 180 / np.pi / 2).astype(np.uint8)
            self.mask[..., 1] = 255
            self.mask[..., 2] = cv.normalize(res_magnitude, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
            rgb = cv.cvtColor(self.mask, cv.COLOR_HSV2BGR)

            bounding_boxes = self.__pixel_clustering__(
                residual_flow_x, residual_flow_y,
                frame_for_drawing=frame,  # draw boxes on the current downscaled color frame
                mag_threshold=2.0,
                size_threshold=10,
                num_bins=36
            )

            merged_bboxes = self.__filter_and_merge_bounding_boxes__(bounding_boxes, min_l=10, max_l= 30)

            # 3) Draw the final bounding boxes on 'rgb' or 'frame'
            for (b_id, (x1, y1, x2, y2)) in merged_bboxes:
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Show frames if desired
            cv.imshow('Input (Downsized)', frame)
            cv.imshow('Residual Flow', rgb)

            # Write outputs if needed

            self.__write_results__(rgb, gray)
            

            # Update variables for next iteration
            self.prev_gray = gray
            self.prev_raw = raw_frame

            # Handle "inspect frames" or "press Q to quit"
            if self.INSPECT_FRAMES:
                print("Press Enter to see next frame (or Esc to exit)...")
                while True:
                    key = cv.waitKey(0) & 0xFF
                    if key == 13:  # Enter
                        break
                    elif key == 27:  # Escape
                        self.__cleanup__()
                        exit()
            else:
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

        self.__cleanup__()

    def __write_results__(self, rgb, gray):
        if self.out_optical is not None:
            self.out_optical.write(rgb)
        if self.out_original is not None:
            # If you want color, write the raw_frame. If you want grayscale, do gray => BGR
            # E.g., write the downsize grayscale as color
            gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
            self.out_original.write(gray_bgr)


    def __cleanup__(self):
        """Release resources and close windows."""
        if self.cap is not None:
            self.cap.release()
        if self.out_optical is not None:
            self.out_optical.release()
        if self.out_original is not None:
            self.out_original.release()
        cv.destroyAllWindows()

    def run_tracker(self, video_path):
        """Orchestrates the entire process: init and track."""
        self.__init_video__(video_path)
        self.__track_projectiles__()


        
        
if __name__ == '__main__':
    tracker = ProjectileTracker()
    tracker.run_tracker('data/movement.mp4')