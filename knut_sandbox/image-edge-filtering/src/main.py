import cv2
import time
import os
from filters import apply_filter_1, apply_filter_2, apply_filter_3

def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    return cv2.imread(image_path)

def ensure_output_dir(path='output'):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def main():
    image_path = 'knut_sandbox/testing_folder/test_image.png'
    image = load_image(image_path)

    filters = [
        (apply_filter_1, {'param1': 1, 'param2': 2}),
        (apply_filter_2, {'param1': 3}),
        (apply_filter_3, {'param1': 3, 'param2': 5, 'param3': 0}),
    ]
    
    output_dir = ensure_output_dir()

    for filter_func, params in filters:
        start_time = time.time()
        processed_image = filter_func(image, **params)
        processing_time = time.time() - start_time
        print(f"Processed with {filter_func.__name__} in {processing_time:.4f} seconds")
        
        # Save the processed image
        output_path = os.path.join(output_dir, f"{filter_func.__name__}.png")
        cv2.imwrite(output_path, processed_image)
        print(f"Saved {output_path}")
        
        # Display the processed image in a window
        cv2.imshow(f"{filter_func.__name__}", processed_image)
        cv2.waitKey(0)  # Wait until a key is pressed
        cv2.destroyWindow(f"{filter_func.__name__}")

if __name__ == "__main__":
    main()