from PIL import Image, ImageDraw
import os

if not os.path.exists('cropped'):
    os.makedirs('cropped')

image_path = 'frames/frame_7.jpg'
original_image = Image.open(image_path)



# Parameters for the new square image and bounding box coordinates
square_size = 500

# Parameters for the new square image, bounding box coordinates, and padding
padding = 50  # Padding around the bounding box
xtl, ytl, xbr, ybr = 1053.22, 257.82, 1226.26, 336.11

# Calculate the bounding box's width and height
bbox_center = ((xtl + xbr) / 2, (ytl + ybr) / 2)

bbox_width = xbr - xtl
bbox_height = ybr - ytl

# Determine the size of the square based on the larger dimension of the bounding box and padding
square_size = max(bbox_width, bbox_height) + 2 * padding

# Load the original image
original_image = Image.open(image_path)

# Create a new square image with a white background
new_image = Image.new('RGB', (int(square_size), int(square_size)), (255, 255, 255))

# Calculate the new bounding box's position in the square image, ensuring it is centered
new_bbox_x = (square_size - bbox_width) / 2
new_bbox_y = (square_size - bbox_height) / 2

# Calculate the crop box coordinates, ensuring we do not go out of bounds
crop_left = max(bbox_center[0] - (square_size / 2), 0)
crop_upper = max(bbox_center[1] - (square_size / 2), 0)
crop_right = min(crop_left + square_size, original_image.width)
crop_lower = min(crop_upper + square_size, original_image.height)

cropped_area = original_image.crop((int(crop_left), int(crop_upper), int(crop_right), int(crop_lower)))

# Resize the cropped area to fit the square size if it's larger than the intended square
if cropped_area.width > square_size or cropped_area.height > square_size:
    cropped_area = cropped_area.resize((int(square_size), int(square_size)), Image.Resampling.LANCZOS)

# Calculate the position to paste the cropped area onto the new square image
paste_x = (new_image.width - cropped_area.width) // 2
paste_y = (new_image.height - cropped_area.height) // 2

# Paste the cropped area onto the new square image
new_image.paste(cropped_area, (paste_x, paste_y))

# Draw the bounding box on the new image
draw = ImageDraw.Draw(new_image)
draw.rectangle([(paste_x + new_bbox_x, paste_y + new_bbox_y), 
                (paste_x + new_bbox_x + bbox_width, paste_y + new_bbox_y + bbox_height)], 
                outline="red", width=2)

# Save the new square image with the object and padding
output_path_with_object_and_padding = 'cropped/square_frame_7.jpg'
new_image.save(output_path_with_object_and_padding)

output_path_with_object_and_padding
