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


class Projectile:
    def __init__(self, bbox, bin_id, flow_vector):
        self.id = None  #Only assign id's to confirmed projectiles.
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.bin_id = bin_id  # motion direction bin
        self.flow_vector = flow_vector  # (vx, vy)
        self.age = 0    #Number of frames survived.
        self.missed = 0
        self.confirmed = False
        

class ProjectileTracker:
    def __init__(self, debugging=False):
        # Configuration flags
        self.CREATE_DEMO_MP4 = True
        self.MAGNITUDE_HEATMAP = debugging
        self.OPTICAL_FLOW_WINDOW = debugging
        self.debugging = debugging

        self.INSPECT_FRAMES = debugging

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
        self.pixel_diff_log = []

        # For heatmap visualization
        self.fig = None
        self.ax = None
        self.im = None
        
        #Projectile tracking
        self.projectiles: list[Projectile] = []
        self.next_projectile_id = 0
        self.max_missed = 3

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
        self.num_bins = num_bins
        
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
                if size < size_threshold:   #Ignore too small BB
                    continue

                min_y, max_y = coords[0].min(), coords[0].max()
                min_x, max_x = coords[1].min(), coords[1].max()


                coords_list = list(zip(coords[1], coords[0]))  # (x, y) order
                bounding_boxes.append((b, (min_x, min_y, max_x, max_y), coords_list))

                #bounding_boxes.append((b, (min_x, min_y, max_x, max_y)))   No segmentation mask
                

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
        
        for (b_id, (x1, y1, x2, y2), pixel_coords) in bounding_boxes:
            w, h = (x2 - x1), (y2 - y1)
            if min_l <= w <= max_l and min_l <= h <= max_l:
                filtered_boxes.append((b_id, (x1, y1, x2, y2), pixel_coords))

        # 2) We'll do naive iterative merging of overlapping boxes that share the same bin_id
        merged = True
        while merged:
            merged = False
            result = []
            while filtered_boxes:
                curr_bin, curr_box, curr_coords = filtered_boxes.pop()
                merged_index = None
                for i, (r_bin, r_box, r_coords) in enumerate(result):
                    if self.__bins_are_neighbors__(r_bin, curr_bin) and self.__boxes_overlap__(curr_box, r_box):
                        # Merge boxes
                        new_box = self.__merge_boxes__(curr_box, r_box)
                        # Merge pixel coordinates
                        new_coords = curr_coords + r_coords  # concatenate the two lists
                        # Replace the merged item in result
                        result[i] = (r_bin, new_box, new_coords)
                        merged = True
                        merged_index = i
                        break

                if merged_index is None:
                    # No merge found => keep the current box
                    result.append((curr_bin, curr_box, curr_coords))

            filtered_boxes = result

        return filtered_boxes
    
    
    def __bins_are_neighbors__(self, bin1, bin2):
        diff = abs(bin1 - bin2)
        return diff <= 1 or diff >= (self.num_bins - 1)
    

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
            
            
    def __center__(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    def __estimate_flow__(self, pixel_coords, flow_x, flow_y):
        vx_list = []
        vy_list = []
        for (x, y) in pixel_coords:
            vx_list.append(flow_x[y, x])
            vy_list.append(flow_y[y, x])
        if vx_list:
            return np.mean(vx_list), np.mean(vy_list)
        else:
            return 0.0, 0.0
        
        
    def __update_projectiles__(self, detections, residual_flow_x, residual_flow_y):
        predictions = []
        for projectile in self.projectiles:
            cx, cy = self.__center__(projectile.bbox)
            vx, vy = projectile.flow_vector
            pred_cx = cx + vx
            pred_cy = cy + vy
            predictions.append((projectile, (pred_cx, pred_cy)))

        assigned_projectiles = set()
        assigned_detections = set()

        for i, (bin_id, (x1, y1, x2, y2), pixel_coords) in enumerate(detections):
            cx_det, cy_det = self.__center__((x1, y1, x2, y2))
            best_projectile = None
            best_distance = float('inf')

            for projectile, (pred_cx, pred_cy) in predictions:
                dist = np.hypot(pred_cx - cx_det, pred_cy - cy_det)
                bin_diff = min(abs(projectile.bin_id - bin_id), self.num_bins - abs(projectile.bin_id - bin_id))

                if dist < 30 and bin_diff <= 2:
                    if dist < best_distance:
                        best_distance = dist
                        best_projectile:Projectile = projectile

            if best_projectile is not None:
                # Update existing projectile
                best_projectile.bbox = (x1, y1, x2, y2)
                mean_vx, mean_vy = self.__estimate_flow__(pixel_coords, residual_flow_x, residual_flow_y)
                best_projectile.flow_vector = (mean_vx, mean_vy)
                best_projectile.bin_id = bin_id
                best_projectile.age += 1
                
                if best_projectile.age >= 3 and not best_projectile.confirmed:
                    best_projectile.confirmed = True
                    best_projectile.id = self.next_projectile_id
                    self.next_projectile_id += 1
                    
                best_projectile.missed = 0
                assigned_projectiles.add(best_projectile)
                assigned_detections.add(i)

        # Create new tracks for unmatched detections
        for i, (bin_id, (x1, y1, x2, y2), pixel_coords) in enumerate(detections):
            if i not in assigned_detections:
                mean_vx, mean_vy = self.__estimate_flow__(pixel_coords, residual_flow_x, residual_flow_y)
                new_projectile = Projectile((x1, y1, x2, y2), bin_id, (mean_vx, mean_vy))
                self.projectiles.append(new_projectile)
                #self.next_track_id += 1

        # Age and remove tracks that are missed too much
        for projectile in self.projectiles:
            if projectile not in assigned_projectiles:
                projectile.missed += 1

        self.projectiles = [p for p in self.projectiles if p.missed <= self.max_missed]

    
    def __track_projectiles__(self):
        """Main loop to read frames, compute flow, subtract background, detect projectiles, etc."""
        while self.cap.isOpened():
            # Skip N frames if needed
            for _ in range(self.N_FRAMES_SKIP + 1):
                ret, frame = self.cap.read()

            if not ret:
                break

            # Check difference from previous frame (to maybe skip near-identical frames)
            frame_diff = cv.absdiff(frame, self.prev_raw)
            max_pixel_diff = np.max(frame_diff)
            
            if self.INSPECT_FRAMES:
                print(f'Diff between current and prior frame: {np.sum(frame_diff)}, Max pixel diff: {max_pixel_diff}')
            self.pixel_diff_log.append(max_pixel_diff)
            if len(self.pixel_diff_log) > 10:
                self.pixel_diff_log.pop(0)

            if self.INSPECT_FRAMES:
                print(max_pixel_diff, np.average(self.pixel_diff_log))
            if max_pixel_diff < np.average(self.pixel_diff_log) / 2:
                if self.INSPECT_FRAMES:
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

            res_magnitude, res_angle = cv.cartToPolar(residual_flow_x, residual_flow_y, angleInDegrees=False)

            

            bounding_boxes = self.__pixel_clustering__(
                residual_flow_x, residual_flow_y,
                frame_for_drawing=frame,  # draw boxes on the current downscaled color frame
                mag_threshold=2.0,
                size_threshold=10,
                num_bins=36
            )

            merged_bboxes = self.__filter_and_merge_bounding_boxes__(bounding_boxes, min_l=0, max_l= 100)
            
            self.__update_projectiles__(merged_bboxes, residual_flow_x, residual_flow_y)
            
            if self.debugging:
                #self.__draw_debug_info__(frame, merged_bboxes, res_magnitude)
                pass


            self.__draw_projectile_bounding_boxes__(frame, res_magnitude)
            
            # Show frames if desired
            if self.debugging:
                cv.imshow('Input (Downsized)', frame)
            
            else:
                # Upsample back to original size
                frame_up = cv.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2),
                                    interpolation=cv.INTER_LINEAR)
                cv.imshow('Input (Upsampled)', frame_up)
                        
            
            if self.OPTICAL_FLOW_WINDOW:
                self.mask[..., 0] = (res_angle * 180 / np.pi / 2).astype(np.uint8)
                self.mask[..., 1] = 255
                self.mask[..., 2] = cv.normalize(res_magnitude, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
                rgb = cv.cvtColor(self.mask, cv.COLOR_HSV2BGR)
                cv.imshow('Residual Flow', rgb)


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
        
    def __draw_projectile_bounding_boxes__(self, frame, res_magnitude):
        """
        Draw merged bounding boxes and, if tracking is active, draw Track IDs.
        
        Parameters:
            frame: The frame to draw on
            merged_bboxes: List of (bin_id, bbox, pixel_coords)
            debugging: If True, draw extra debug info (bin range, mean mag)
        """
        overlay = frame.copy()

        for projectile in self.projectiles:
            if not projectile.confirmed:
                continue
            b_id = projectile.bin_id
            x1, y1, x2, y2 = projectile.bbox
            
            # Draw bounding box
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if self.debugging:
                # Draw bin angle range
                # Draw bin and mean mag as before
                bin_start_deg = b_id * (360 / self.num_bins)
                bin_end_deg = (b_id + 1) * (360 / self.num_bins)
                label_text = f"{int(bin_start_deg)}-{int(bin_end_deg)} Angle"

                roi_magnitude = res_magnitude[y1:y2, x1:x2]
                mean_magnitude = np.mean(roi_magnitude)
                mag_text = f"{mean_magnitude:.2f} Magnitude"

                cv.putText(frame, label_text, (x1, min(frame.shape[0]-1, y2+8)),
                        cv.FONT_HERSHEY_SIMPLEX, 0.2, (0,255,0), 1, cv.LINE_AA)

                cv.putText(frame, mag_text, (x1, min(frame.shape[0]-1, y2+16)),
                        cv.FONT_HERSHEY_SIMPLEX, 0.2, (0,255,0), 1, cv.LINE_AA)
                
        # Now draw the Track IDs separately
        for projectile in self.projectiles:
            if not projectile.confirmed:
                continue
            x1, y1, x2, y2 = projectile.bbox
            projectile_text = f'ID {projectile.id}'

            # Optional: background box for text (easier to read)
            (tw, th), _ = cv.getTextSize(projectile_text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv.rectangle(frame, (x1, y1-20), (x1+tw, y1-5), (255, 0, 0), cv.FILLED)

            # Draw the ID text
            cv.putText(frame, projectile_text, (x1, y1-7),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
            
            
        
    def __draw_debug_info__(self, frame, bounding_boxes, res_magnitude):
        overlay = frame.copy()

        for (b_id, (x1, y1, x2, y2), pixel_coords) in bounding_boxes:

            # Draw segmentation mask: color only the segmented region
            for (x, y) in pixel_coords:
                overlay[y, x] = (0, 0, 255)  # Red color

            # Draw bin and mean mag as before
            bin_start_deg = b_id * (360 / self.num_bins)
            bin_end_deg = (b_id + 1) * (360 / self.num_bins)
            label_text = f"{int(bin_start_deg)}A"

            roi_magnitude = res_magnitude[y1:y2, x1:x2]
            mean_magnitude = np.mean(roi_magnitude)
            mag_text = f"{mean_magnitude:.2f}M"

            cv.putText(frame, label_text, (x1, max(0, y1-10)),
                    cv.FONT_HERSHEY_SIMPLEX, 0.2, (0,255,0), 1, cv.LINE_AA)

            cv.putText(frame, mag_text, (x1, min(frame.shape[0]-1, y2+15)),
                    cv.FONT_HERSHEY_SIMPLEX, 0.2, (0,255,0), 1, cv.LINE_AA)

        # After loop, blend the overlay on top
        alpha = 0.4  # 40% red
        frame[:] = cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

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
        """Orchestrates the entire process: init and track.
           Set debugging to true if you want additional info on the bounding boxes."""
        self.__init_video__(video_path)
        self.__track_projectiles__()


        
        
if __name__ == '__main__':
    tracker = ProjectileTracker(debugging=False)
    video_path = "120fps_data/2025-03-26 10-43-07.mp4"
    tracker.run_tracker(video_path)