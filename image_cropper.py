from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import os
import argparse

parser = argparse.ArgumentParser(description='Process images and annotations for object detection.')
parser.add_argument('target_image_size', type=int, help='Target image size for the square crop.')
parser.add_argument('video_name', type=str, help='Name of the video for processing.')
parser.add_argument('annotation_path', type=str, help='Path to the initial XML annotations file.')

args = parser.parse_args()

target_image_size = args.target_image_size
video_name = args.video_name
annotation_path = args.annotation_path

additional_padding = 10

cropped_folder_path = f'cropped/{video_name}'
os.makedirs(cropped_folder_path, exist_ok=True)

print(f"Processing annotations from: {annotation_path}")

tree = ET.parse(annotation_path)
root = tree.getroot()

for track in root.findall('.//track'):
    if track.attrib.get('label') == 'queen':
        for box in track.findall('box'):
            frame_num = box.attrib['frame']
            print(f"Processing frame: {frame_num}")

            image_path = os.path.join(f'frames/frames_{video_name}/frame_{frame_num}.png')
            if not os.path.exists(image_path):
                continue

            original_image = Image.open(image_path)
            xtl, ytl, xbr, ybr = [float(box.attrib[attr]) for attr in ['xtl', 'ytl', 'xbr', 'ybr']]

            bbox_width = xbr - xtl
            bbox_height = ybr - ytl

            initial_square_size = max(bbox_width, bbox_height)

            predefined_square_size_temp = target_image_size
            if initial_square_size > target_image_size: # if the bbox is larger than the predefined square size
                target_image_size = round(initial_square_size) + additional_padding # adding predefined padding pixels to the square size to make sure the object is not cropped out

            padding_height = (target_image_size - bbox_height) / 2
            padding_width = (target_image_size - bbox_width) / 2

            # the new square crop area values
            crop_left = xtl - padding_width
            crop_upper = ytl - padding_height
            crop_right = xbr + padding_width
            crop_lower = ybr + padding_height

            cropped_area = original_image.crop((int(crop_left), int(crop_upper), int(crop_right), int(crop_lower)))
            new_image = Image.new('RGB', (target_image_size, target_image_size), (255, 255, 255))

            # pasting the cropped image to the new image
            paste_x = (target_image_size - cropped_area.width) // 2
            paste_y = (target_image_size - cropped_area.height) // 2
            new_image.paste(cropped_area, (paste_x, paste_y))

            if predefined_square_size_temp != target_image_size:
                new_image = new_image.resize((predefined_square_size_temp, predefined_square_size_temp))
                resize_ratio = predefined_square_size_temp / target_image_size
                new_tlx = padding_width * resize_ratio
                new_tly = padding_height * resize_ratio
                new_brx = (padding_width * resize_ratio) + (bbox_width * resize_ratio)
                new_bry = (padding_height * resize_ratio) + (bbox_height * resize_ratio)
            else:
                new_tlx = padding_width
                new_tly = padding_height
                new_brx = padding_width + bbox_width
                new_bry = padding_height + bbox_height

            # for new annotation.xml file
            box.set('xtl', str(new_tlx))
            box.set('ytl', str(new_tly))
            box.set('xbr', str(new_brx))
            box.set('ybr', str(new_bry))

            draw = ImageDraw.Draw(new_image)

            # TODO: if you want to check if the bounding boxes within the cropped images are correct, uncomment the
            #  following line draw.rectangle([new_tlx, new_tly, new_brx, new_bry], outline='red', width=2) #test

            new_image.save(f'cropped/{video_name}/cropped_frame{frame_num}.png')
            target_image_size = predefined_square_size_temp

new_xml_path = f'annotations_updated_{video_name}.xml'
tree.write(new_xml_path)
