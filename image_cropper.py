from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import os
import argparse
from pascal_voc import write_pascal_voc

parser = argparse.ArgumentParser(description='Process images and annotations_CVAT for object detection.')
parser.add_argument('target_image_size', type=int, help='Target image size for the square crop.')
parser.add_argument('video_name', type=str, help='Name of the video for processing.')
parser.add_argument('annotation_path', type=str, help='Path to the initial XML annotations_CVAT file.')

args = parser.parse_args()
# TODO: if you don't want to use script with arguments comment the lines above

target_image_size = args.target_image_size
video_name = args.video_name
annotation_path = args.annotation_path

additional_padding = 500 # FOMO needs object to appear smaller

cropped_folder_path = f'cropped/{video_name}/frames_{target_image_size}'
cropped_annotations_folder_path = f'cropped/{video_name}/annotations_{target_image_size}'
os.makedirs(cropped_folder_path, exist_ok=True)
os.makedirs(cropped_annotations_folder_path, exist_ok=True)

print(f"Processing annotations_CVAT from: {annotation_path}")

tree = ET.parse(annotation_path)
root = tree.getroot()

for track in root.findall('.//track'):
    if track.attrib.get('label') == 'queen':
        for box in track.findall('box'):
            frame_num = box.attrib['frame']
            print(f"Processing frame: {frame_num}")

            image_path = os.path.join(f'frames/{video_name}/frame_{frame_num}.png')
            if not os.path.exists(image_path):
                continue

            if frame_num == '263':
                print('debug')

            original_image = Image.open(image_path)
            xtl, ytl, xbr, ybr = [float(box.attrib[attr]) for attr in ['xtl', 'ytl', 'xbr', 'ybr']]

            bbox_width = xbr - xtl
            bbox_height = ybr - ytl

            initial_square_size = max(bbox_width, bbox_height)

            predefined_square_size_temp = target_image_size
            if initial_square_size > target_image_size or initial_square_size > target_image_size / 2 + 40: # if the object is too big, we need to resize it, fomo does not do well with large objects
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
                new_xtl = padding_width * resize_ratio
                new_ytl = padding_height * resize_ratio
                new_xbr = (padding_width * resize_ratio) + (bbox_width * resize_ratio)
                new_ybr = (padding_height * resize_ratio) + (bbox_height * resize_ratio)
            else:
                new_xtl = padding_width
                new_ytl = padding_height
                new_xbr = padding_width + bbox_width
                new_ybr = padding_height + bbox_height

            # for new annotation.xml file
            box.set('xtl', str(new_xtl))
            box.set('ytl', str(new_ytl))
            box.set('xbr', str(new_xbr))
            box.set('ybr', str(new_ybr))

            draw = ImageDraw.Draw(new_image)

            write_pascal_voc(f'{cropped_annotations_folder_path}/frame_{frame_num}.xml', f'frame_{frame_num}.png', predefined_square_size_temp, new_xtl, new_ytl, new_xbr, new_ybr)

            # TODO: if you want to check if the bounding boxes within the cropped images are correct, uncomment the
            #  following line
            #draw.rectangle([new_xtl, new_ytl, new_xbr, new_ybr], outline='red', width=2) #test

            new_image.save(f'{cropped_folder_path}/frame_{frame_num}.png')
            target_image_size = predefined_square_size_temp

new_xml_path = os.path.join(cropped_folder_path, 'annotations.xml')
tree.write(new_xml_path)
