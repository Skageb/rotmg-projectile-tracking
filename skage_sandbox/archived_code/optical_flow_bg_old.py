import cv2 as cv
import numpy as np

CREATE_DEMO_MP4 = True
RESULT_FILE = 'optical_flow/OF_3rd_frame.mp4'
RESULT_ORIGINAL = 'optical_flow/OF_3rd_frame_original.mp4'

N_FRAMES_SKIP = 0

def gaussian_downsize(frame):
    # Apply Gaussian blur with a 5x5 kernel (adjust kernel size and sigma as needed)
    blurred_frame = cv.GaussianBlur(frame, (5, 5), 0)

    # Downsize the image to half its width and height using INTER_AREA interpolation
    frame = cv.resize(blurred_frame, (frame.shape[1] // 2, frame.shape[0] // 2), interpolation=cv.INTER_AREA)
    return frame


#Read in video
cap = cv.VideoCapture('data/movement.mp4')

ret, first_frame = cap.read()

first_frame = gaussian_downsize(first_frame)


mask = np.zeros_like(first_frame)

prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY) 

#Set saturation to max
mask[...,1] = 255

frame_height, frame_width = first_frame.shape[:2]
print(frame_height, frame_width)
prev_frame = first_frame

#
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out_optical = cv.VideoWriter(f'results/{RESULT_FILE}', fourcc, 5.0, (frame_width, frame_height))
out_original = cv.VideoWriter(f'results/{RESULT_ORIGINAL}', fourcc, 5.0, (frame_width, frame_height))


while(cap.isOpened()):
    for n in range(N_FRAMES_SKIP+1):  #Skip N frames
        ret, frame = cap.read()
        
    if not ret:
        break  # No more frames, exit the inner loop

    frame = gaussian_downsize(frame)

    cv.imshow("input", frame)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.8, 3, 25, 3, 5, 1.2, 0)

    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

    #Set Hue according to the optical flow direction
    mask[..., 0] = angle*180 / np.pi / 2

    #Sets image value according to normalized 
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

    count_obj = np.unique(mask[...,0], return_counts=True)

    directions, counts = count_obj
    
    most_frequent = directions[np.argmax(counts)]
   

    background_magnitudes = mask[np.where(mask[..., 0] == most_frequent)][...,2]
    
    magnitudes, mcounts = np.unique(background_magnitudes, return_counts=True)
    
    most_frequent_mag = magnitudes[np.argmax(mcounts)]

    bg_angle = (most_frequent * 2) * np.pi / 180
    bg_flow_x = most_frequent_mag * np.cos(bg_angle)
    bg_flow_y = most_frequent_mag * np.sin(bg_angle)

    residual_flow = flow.copy()
    residual_flow[..., 0] -= bg_flow_x
    residual_flow[..., 1] -= bg_flow_y

    residual_magnitude, residual_angle = cv.cartToPolar(residual_flow[..., 0], residual_flow[..., 1])
    # For display in HSV (remember Hue in OpenCV is in the range [0,180]):
    mask[..., 0] = residual_angle * 180 / np.pi / 2
    mask[..., 2] = cv.normalize(residual_magnitude, None, 0, 255, cv.NORM_MINMAX)

    print(f'Background motion, Angle: {most_frequent}, Magnitude {most_frequent_mag}')

    rgb = cv.cvtColor(mask, cv.COLOR_HSV2RGB)

    cv.imshow("Dense optical flow", rgb)

    prev_frame = frame

    prev_gray = gray

    out_optical.write(rgb)
    out_original.write(gray)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

out_optical.release()
out_original.release()
cap.release()
cv.destroyAllWindows()