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

    # Get the bounding box of the largest contour to remove white padding
    x, y, w, h = cv2.boundingRect(contours[0])
    cropped_image = image.crop((x, y, x + w, y + h))

    return cropped_image

def detect_canny_edges(image):
    # Convert the PIL image to a numpy array
    image_np = np.array(image)
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, threshold1=50, threshold2=400)
    
    return edges

def merge_overlapping_contours(contours):
    # Sort contours by area (largest to smallest)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Initialize list to store merged contours
    merged_contours = []

    # Iterate over each contour
    for contour in contours:
        if len(merged_contours) == 0:
            merged_contours.append(contour)
        else:
            # Check if current contour overlaps with the last merged contour
            last_contour = merged_contours[-1]
            
            # Extract the point as a tuple (x, y) from the first point of the contour
            point = tuple(contour[0][0])  # contour[0][0] is a numpy array, convert to tuple
            
            merged_contours.append(contour)
            # # Use pointPolygonTest to check if a contour point is inside the previous contour
            # if cv2.pointPolygonTest(last_contour, point, False) >= 0:
            #     # Merge contours using convex hull (if they are close)
            #     hull = cv2.convexHull(np.concatenate([last_contour, contour], axis=0))
            #     merged_contours[-1] = hull
            # else:
            #     # Add the current contour to the list if no overlap
            #     merged_contours.append(contour)

    return merged_contours

def process_images(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process each image in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.png')):  # Process only .jpg or .png files
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)

            # Detect Canny edges in the image
            edges = detect_canny_edges(image)
            
            # Find contours from the edge-detected image
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Merge overlapping contours
            merged_contours = merge_overlapping_contours(contours)

            # Create an empty black mask image
            mask = np.zeros_like(edges)  # Same size as edges, all pixels are 0

            # Draw the merged contours on the mask (in white color, thickness -1 to fill)
            cv2.drawContours(mask, merged_contours, -1, (255), thickness=cv2.FILLED)

            # Save the mask image
            mask_image = Image.fromarray(mask)
            final_output_path = os.path.join(output_folder, 'mask_' + filename)
            mask_image.save(final_output_path)

            print(f"Processed {filename}, saved mask as {final_output_path}")

# Example usage
input_folder = 'processed'  # Path to your input folder
output_folder = 'output_images'  # Path to your output folder

process_images(input_folder, output_folder)
