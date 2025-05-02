import cv2
import numpy as np


def apply_sobel_filter(image, ksize=3):
    import cv2
    import time

    start_time = time.time()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=ksize)
    sobel_image = cv2.magnitude(sobel_x, sobel_y)
    processing_time = time.time() - start_time

    return sobel_image, processing_time


def apply_canny_filter(image, threshold1=100, threshold2=200):
    import cv2
    import time

    start_time = time.time()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny_image = cv2.Canny(gray_image, threshold1, threshold2)
    processing_time = time.time() - start_time

    return canny_image, processing_time


def apply_laplacian_filter(image, ksize=3):
    import cv2
    import time

    start_time = time.time()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_image = cv2.Laplacian(gray_image, cv2.CV_64F, ksize=ksize)
    processing_time = time.time() - start_time

    return laplacian_image, processing_time

def apply_filter_1(image, param1, param2):
    """
    Applies a Canny edge detector.
    Using param1 and param2 to derive lower and upper thresholds.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lower_threshold = param1 * 250      # Example adjustment
    upper_threshold = param2 * 255
    edges = cv2.Canny(gray, lower_threshold, upper_threshold)
    # Convert single channel edge image back to BGR for consistency
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def apply_filter_2(image, param1):
    """
    Applies a Sobel edge detector.
    'param1' is used as the kernel size.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute gradients along the x and y axis, using kernel size (must be odd)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=param1)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=param1)
    sobel = cv2.magnitude(sobelx, sobely)
    # Normalize the result to 0-255 then convert back to uint8
    sobel = np.uint8(255 * sobel / np.max(sobel))
    return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)

def apply_filter_3(image, param1, param2, param3):
    """
    Applies a Laplacian edge detector.
    'param1' is the ksize (must be odd), param2 and param3 can be used for additional processing if needed.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=param1)
    laplacian = cv2.convertScaleAbs(laplacian)
    return cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)