import cv2
import os
import xml.etree.ElementTree as ET
from pascal_voc import write_pascal_voc
from PIL import Image, ImageDraw

target_image_width = 320
target_image_height = 320

compressed_folder_path = f'compressed/micro_all_frames_training/images'
compressed_annotations_folderpath = f'compressed/micro_all_frames_training/annotations'
os.makedirs(compressed_folder_path, exist_ok=True)
os.makedirs(compressed_annotations_folderpath, exist_ok=True)


def process(annotation_path, video_name, frame_counter):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    for track in root.findall('.//track'):
        if track.attrib.get('label') == 'queen':
            for box in track.findall('box'):
                frame_num = box.attrib['frame']
                print(f"Processing frame: {frame_counter}")

                image_path = os.path.join(f'frames/{video_name}/frame_{frame_num}.png')
                if not os.path.exists(image_path):
                    continue

                original_image = cv2.imread(image_path)
                xtl, ytl, xbr, ybr = [float(box.attrib[attr]) for attr in ['xtl', 'ytl', 'xbr', 'ybr']]
                resized_image = cv2.resize(original_image, (target_image_width, target_image_height))

                cv2.imwrite(f'{compressed_folder_path}/frame_{frame_counter}.png', resized_image)
                print(f"Image saved to: {compressed_folder_path}/frame_{frame_counter}.png")

                resize_ratio = target_image_width / original_image.shape[1]

                new_xtl = xtl * resize_ratio
                new_ytl = ytl * resize_ratio
                new_xbr = xbr * resize_ratio
                new_ybr = ybr * resize_ratio

                write_pascal_voc(f'{compressed_annotations_folderpath}/frame_{frame_counter}.xml',
                                 f'frame_{frame_counter}.png', target_image_width, target_image_height, new_xtl,
                                 new_ytl, new_xbr, new_ybr)
                frame_counter += 1

    return frame_counter


frame_counter = 1

for file in os.listdir('videos/'):
    if file.endswith('.webm') and not file.endswith('6.webm') and not file.endswith('11.webm') and not file.endswith('12.webm') and not file.endswith('13.webm') and not file.endswith('14.webm') and not file.endswith('15.webm') and not file.endswith('16.webm') and not file.endswith('17.webm'):
        video_name = file.split('.')[0]
        annotation_num = video_name[-2:] if video_name[-2:].isdigit() else video_name[-1]
        current_annotation = f'annotations_CVAT/micro{annotation_num}.xml'
        frame_counter = process(current_annotation, video_name, frame_counter)  # Update frame_counter with the returned value
