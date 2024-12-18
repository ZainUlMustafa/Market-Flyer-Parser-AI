import cv2

# Image coordinates and product details
products = [
    {"name": "Keurig K-Express Essentials Single-Serve Coffee Maker", "coords": (70, 160, 100, 200)},
    {"name": "Apple Watch SE (2nd generation, GPS, 40mm)", "coords": (480, 160, 120, 120)},
    {"name": "Magic Bullet 7-Piece Personal Blender", "coords": (50, 400, 120, 220)},
    {"name": "TCL 55\" Q Class 4K UHD HDR QLED Smart Google TV", "coords": (350, 400, 150, 120)},
    {"name": "Ninja XL 5.5-qt Air Fryer", "coords": (50, 650, 140, 200)},
    {"name": "Nintendo Switch OLED Mario Kart 8 Deluxe Bundle", "coords": (450, 650, 150, 220)}
]

# Load the image you want to overlay the bounding boxes on
image_path = "processed/data.png"  # Change this to your image path
image = cv2.imread(image_path)

# Loop over the products to draw the bounding boxes
for product in products:
    x, y, w, h = product["coords"]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green bounding box

    # Optionally, add text for the product name
    cv2.putText(image, product["name"], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Save or display the final image with bounding boxes
output_path = "output_image_with_bboxes.jpg"  # Change this to your desired output path
cv2.imwrite(output_path, image)

# Optionally, display the image (useful for debugging)
cv2.imshow("Image with Bounding Boxes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
