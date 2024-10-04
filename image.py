import cv2
import os
import numpy as np

# Function to apply image processing based on the settings
def process_image(image, hmin, hmax, smin, smax, vmin, vmax, 
                  gaussian_blur, canny_low, canny_high, 
                  kernel_d, kernel_e, dilation, erosion):
    
    # Convert image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Apply HSV thresholding
    lower_bound = np.array([hmin, smin, vmin])
    upper_bound = np.array([hmax, smax, vmax])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    result = cv2.bitwise_and(image, image, mask=mask)

    # Apply Gaussian Blur
    if gaussian_blur > 0:
        result = cv2.GaussianBlur(result, (gaussian_blur*2+1, gaussian_blur*2+1), 0)

    # Apply Canny Edge Detection
    edges = cv2.Canny(result, canny_low, canny_high)

    # Create kernels for dilation and erosion
    kernel_d = np.ones((kernel_d, kernel_d), np.uint8)
    kernel_e = np.ones((kernel_e, kernel_e), np.uint8)

    # Apply Dilation
    if dilation > 0:
        edges = cv2.dilate(edges, kernel_d, iterations=dilation)

    # Apply Erosion
    if erosion > 0:
        edges = cv2.erode(edges, kernel_e, iterations=erosion)

    return edges

# Function to process a single image and save it to a specified folder
def process_single_image(input_file, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Set the values based on your configuration
    hmin, hmax = 0, 111
    smin, smax = 0, 213
    vmin, vmax = 129, 239
    gaussian_blur = 2
    canny_low, canny_high = 251, 179
    kernel_d, kernel_e = 9, 12  # Kernel sizes for dilation and erosion
    dilation, erosion = 0, 0

    # Read the input image
    image = cv2.imread(input_file)

    if image is not None:
        # Process the image with the defined settings
        processed_image = process_image(image, hmin, hmax, smin, smax, vmin, vmax,
                                        gaussian_blur, canny_low, canny_high,
                                        kernel_d, kernel_e, dilation, erosion)

        # Save the processed image in the output folder with the same filename
        filename = os.path.basename(input_file)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, processed_image)

        print(f"Processed and saved: {output_path}")
    else:
        print(f"Could not read image: {input_file}")

# Paths for the input file and output folder
input_file = '/Users/user/Desktop/ty/ADELINE LEE XI YEAN_page_1_img_1.jpeg'  # Replace with your image path
output_folder = '/Users/user/Desktop/TY/processed' # Replace with your folder path

# Process the single image
process_single_image(input_file, output_folder)
