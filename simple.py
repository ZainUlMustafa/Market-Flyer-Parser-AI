import cv2
import numpy as np
from PIL import Image
import os

def remove_white_padding(image):
    # Convert the PIL image to a numpy array
    image_np = np.array(image)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Apply threshold to create a binary image, where white becomes 255 and others become 0
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image (in green color, thickness 3)
    cv2.drawContours(image_np, contours, -1, (0, 255, 0), 3)

    # Get the bounding box of the largest contour to remove white padding
    x, y, w, h = cv2.boundingRect(contours[0])
    print(x, y, w, h)
    cropped_image = image.crop((x, y, x + w, y + h))
    
    # Show contours on the image
    # cv2.imshow("Contours", image_np)
    # cv2.waitKey(0)  # Wait until a key is pressed
    # cv2.destroyAllWindows()

    return cropped_image

def detect_canny_edges(image):
    # Convert the PIL image to a numpy array
    image_np = np.array(image)
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, threshold1=50, threshold2=400)
    
    # Convert edges back to a PIL image
    edges_image = Image.fromarray(edges)
    
    return edges_image

def process_images(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process each image in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.png')):  # Process only .jpg or .png files
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)

            # image_no_padding = remove_white_padding(image)

            # Detect Canny edges in the image
            edges_image = detect_canny_edges(image)

            # Save the processed image
            final_output_path = os.path.join(output_folder, 'edges_' + filename)
            edges_image.save(final_output_path)

            print(f"Processed {filename}, saved as {final_output_path}")

# Example usage
input_folder = 'processed'  # Path to your input folder
output_folder = 'output_images'  # Path to your output folder

process_images(input_folder, output_folder)
