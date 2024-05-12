import cv2
import os
import xml.etree.ElementTree as ET
from pascal_voc import write_pascal_voc
from PIL import Image, ImageDraw
import shutil

target_image_width = 320
target_image_height = 320

compressed_folder_path = f'compressed/micro_all_frames/images'
compressed_annotations_folderpath = f'compressed/micro_all_frames/annotations'
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
    if file.endswith():
        video_name = file.split('.')[0]
        annotation_num = video_name[-2:] if video_name[-2:].isdigit() else video_name[-1]
        current_annotation = f'annotations_CVAT/micro{annotation_num}.xml'
        frame_counter = process(current_annotation, video_name, frame_counter)  # Update frame_counter with the returned value

for i in range(1, 6):
    os.makedirs(f'compressed/micro_all_frames{i}/training/images', exist_ok=True)
    os.makedirs(f'compressed/micro_all_frames{i}/training/annotations', exist_ok=True)
    os.makedirs(f'compressed/micro_all_frames{i}/testing/images', exist_ok=True)
    os.makedirs(f'compressed/micro_all_frames{i}/testing/annotations', exist_ok=True)

# create 5 splits of the dataset, 80% training, 20% testing and put them in the respective folders

for i in range(1, 6):
    for file in os.listdir(compressed_folder_path):
        if file.endswith('.png'):
            image_path = f'{compressed_folder_path}/{file}'
            annotation_path = f'{compressed_annotations_folderpath}/{file.split(".")[0]}.xml'
            if int(file.split('_')[1].split('.')[0]) % 5 == i:
                # copy file to testing
                shutil.copy(image_path, f'compressed/micro_all_frames{i}/testing/images/{file}')
                shutil.copy(annotation_path, f'compressed/micro_all_frames{i}/testing/annotations/{file.split(".")[0]}.xml')
            else:
                shutil.copy(image_path, f'compressed/micro_all_frames{i}/training/images/{file}')
                shutil.copy(annotation_path, f'compressed/micro_all_frames{i}/training/annotations/{file.split(".")[0]}.xml')
