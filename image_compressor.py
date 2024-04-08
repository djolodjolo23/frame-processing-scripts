
import cv2
import os
import xml.etree.ElementTree as ET
from pascal_voc import write_pascal_voc
from PIL import Image, ImageDraw



target_image_width = 1280 #maintain 4:3 ratio
target_image_height = 720 # maitain 4:3 ratio
video_name = 'GX0110088'
annotation_path = 'annotations_CVAT/annotations.xml'


compressed_folder_path = f'compressed/{video_name}/frames_{target_image_width} x {target_image_height}'
compressed_annotations_folderpath = f'compressed/{video_name}/annotations_{target_image_width} x {target_image_height}'
os.makedirs(compressed_folder_path, exist_ok=True)
os.makedirs(compressed_annotations_folderpath, exist_ok=True)

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

            original_image = cv2.imread(image_path)
            xtl, ytl, xbr, ybr = [float(box.attrib[attr]) for attr in ['xtl', 'ytl', 'xbr', 'ybr']]

            bbox_width = xbr - xtl
            bbox_height = ybr - ytl

            resized_image = cv2.resize(original_image, (target_image_width, target_image_height))

            cv2.imwrite(f'{compressed_folder_path}/frame_{frame_num}.png', resized_image)
            print(f"Image saved to: {compressed_folder_path}/frame_{frame_num}.png")

            resize_ratio = target_image_width / original_image.shape[1]

            new_xtl = xtl * resize_ratio
            new_ytl = ytl * resize_ratio
            new_xbr = xbr * resize_ratio
            new_ybr = ybr * resize_ratio

            write_pascal_voc(f'{compressed_annotations_folderpath}/frame_{frame_num}.xml', f'frame_{frame_num}.png', target_image_width, target_image_height, new_xtl, new_ytl, new_xbr, new_ybr)

            
            