import cv2 as cv
import numpy as np

CREATE_DEMO_MP4 = True
RESULT_FILE = 'optical_flow/OF_3rd_frame.mp4'
RESULT_ORIGINAL = 'optical_flow/OF_3rd_frame_original.mp4'

N_FRAMES_SKIP = 0
INSPECT_FRAMES = True

def gaussian_downsize(frame):
    # Apply Gaussian blur with a 5x5 kernel (adjust kernel size and sigma as needed)
    blurred_frame = cv.GaussianBlur(frame, (5, 5), 0)

    # Downsize the image to half its width and height using INTER_AREA interpolation
    frame = cv.resize(blurred_frame, (frame.shape[1] // 2, frame.shape[0] // 2), interpolation=cv.INTER_AREA)
    return frame


#Read in video
cap = cv.VideoCapture('data/movement.mp4')


ret, first_frame = cap.read()

raw_frame = first_frame

first_frame = gaussian_downsize(first_frame)

mask = np.zeros_like(first_frame)

prev_raw = raw_frame
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY) 

#Set saturation to max
mask[...,1] = 255

frame_height, frame_width = first_frame.shape[:2]


#
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out_optical = cv.VideoWriter(f'results/{RESULT_FILE}', fourcc, 5.0, (frame_width, frame_height))
out_original = cv.VideoWriter(f'results/{RESULT_ORIGINAL}', fourcc, 5.0, (frame_width, frame_height))


while(cap.isOpened()):
    for n in range(N_FRAMES_SKIP+1):  #Skip N frames
        ret, frame = cap.read()
        
    if not ret:
        break  # No more frames, exit the inner loop

    raw_frame = frame

    diff = cv.absdiff(raw_frame, prev_raw)
    #print('Diff current and previous frame', np.sum(diff))
    if np.sum(diff) < 600000:
        continue
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

    print(f'Background motion, Angle: {most_frequent}, Magnitude {most_frequent_mag}')

    bg_angle = (most_frequent * 2) * np.pi / 180

    bg_flow_x = most_frequent_mag * np.cos(bg_angle)
    bg_flow_y = most_frequent_mag * np.sin(bg_angle)


    x_mag = np.cos(mask[..., 2])
    y_mag = np.sin(mask[...,2])
    angles = mask[...,0]*2

    x_rel = x_mag - bg_flow_x
    y_rel = y_mag - bg_flow_y
    angle = np.arctan(y_rel/x_rel)
    mask[..., 0] = angle*180 / np.pi / 2
    mask[..., 2] = x_rel / np.cos(angle)


    print(f'Background motion, Angle: {most_frequent}, Magnitude {most_frequent_mag}')



    rgb = cv.cvtColor(mask, cv.COLOR_HSV2RGB)

    cv.imshow("Dense optical flow", rgb)

    prev_frame = frame
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

out_optical.release()
out_original.release()
cap.release()
cv.destroyAllWindows()