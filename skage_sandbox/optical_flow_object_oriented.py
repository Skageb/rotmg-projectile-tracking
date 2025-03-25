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