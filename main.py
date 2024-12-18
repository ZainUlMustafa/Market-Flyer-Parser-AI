import numpy as np
import os
from PIL import Image
import torch
from torchvision import transforms
from u2net import utils, model
import cv2

# Initialize the U-2-Net model
model_path = 'u2net.pth'  # Ensure that you have the model path correct
model_pred = model.U2NET(3, 1)
model_pred.load_state_dict(torch.load(model_path, map_location="cpu"))
model_pred.eval()

# Normalize the prediction
def norm_pred(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn

# Preprocess the image before feeding it into the model
def preprocess(image):
    label_3 = np.zeros(image.shape)
    label = np.zeros(label_3.shape[0:2])

    if 3 == len(label_3.shape):
        label = label_3[:, :, 0]
    elif 2 == len(label_3.shape):
        label = label_3

    if 3 == len(image.shape) and 2 == len(label.shape):
        label = label[:, :, np.newaxis]
    elif 2 == len(image.shape) and 2 == len(label.shape):
        image = image[:, :, np.newaxis]
        label = label[:, :, np.newaxis]

    transform = transforms.Compose([utils.RescaleT(320), utils.ToTensorLab(flag=0)])
    sample = transform({"imidx": np.array([0]), "image": image, "label": label})

    return sample

# Function to remove the background using U-2-Net
def remove_bg(image):
    sample = preprocess(np.array(image))

    with torch.no_grad():
        inputs_test = torch.FloatTensor(sample["image"].unsqueeze(0).float())

        # Perform inference
        d1, _, _, _, _, _, _ = model_pred(inputs_test)
        pred = d1[:, 0, :, :]
        predict = norm_pred(pred).squeeze().cpu().detach().numpy()
        img_out = Image.fromarray(predict * 255).convert("RGB")
        img_out = img_out.resize((image.size), resample=Image.BILINEAR)
        empty_img = Image.new("RGBA", (image.size), 0)
        img_out = Image.composite(image, empty_img, img_out.convert("L"))
        del d1, pred, predict, inputs_test, sample

        return img_out

# Function to remove the background for multiple passes
def remove_bg_mult(image):
    img_out = image.copy()
    for _ in range(4):  # Multiple passes to refine background removal
        img_out = remove_bg(img_out)

    img_out = img_out.resize((image.size), resample=Image.BILINEAR)
    empty_img = Image.new("RGBA", (image.size), 0)
    img_out = Image.composite(image, empty_img, img_out)
    return img_out

# Function to change the background to a black one
def change_background(image, background_color=(255, 255, 255)):
    background = Image.new("RGB", image.size, background_color)  # Black background
    img_out = Image.alpha_composite(background.convert("RGBA"), image.convert("RGBA"))
    return img_out

# Processing all images in a folder
input_folder = 'processed'  # Path to your input folder
output_folder = 'output'  # Path to your output folder

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.png')):  # Process only .jpg or .png files
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)

        # Step 1: Remove white padding
        # image_no_padding = remove_white_padding(image)

        # Step 2: Remove background
        image_no_bg = remove_bg(image)

        # Step 3: Replace background with black
        final_image = change_background(image_no_bg)

        final_output_path = os.path.join(output_folder, 'final_' + filename)
        final_image_rgb = final_image.convert("RGB")
        final_image_rgb.save(final_output_path)

        print(f"Processed {filename}, saved as {final_output_path}")

print("Processing completed.")
