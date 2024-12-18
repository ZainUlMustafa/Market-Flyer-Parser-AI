import os
import cv2  # OpenCV library

# Paths to input and output directories
input_folder = 'chat_input'
output_folder = 'chat_output'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    # Create the full path to the image file
    input_path = os.path.join(input_folder, filename)
    
    # Check if it's a file and has a valid image extension
    if os.path.isfile(input_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        
        # Read the image
        image = cv2.imread(input_path)
        
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize the image to 480p resolution while maintaining aspect ratio
        height, width = gray_image.shape[:2]
        aspect_ratio = width / height
        new_height = 1000
        new_width = int(aspect_ratio * new_height)
        gray_image_resized = cv2.resize(gray_image, (new_width, new_height))
        
        # Define the output file path with .jpeg extension
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '_compressed.jpeg')
        
        # Save the image with lower quality (use '95' for high quality, lower values for lower quality)
        cv2.imwrite(output_path, gray_image_resized, [cv2.IMWRITE_JPEG_QUALITY, 70])  # 50 is the quality (0-100)
        
        print(f'Processed and saved: {output_path}')

print("Image processing complete!")
