import os
import cv2
import albumentations as A
from albumentations.augmentations import transforms
from lxml import etree
from shutil import copyfile
import random
import numpy as np


def read_xml(file_path):
    tree = etree.parse(file_path)
    root = tree.getroot()
    boxes = []
    for member in root.findall('object'):
        bndbox = member.find('bndbox')
        boxes.append([
            round(float(bndbox.find('xmin').text)),
            round(float(bndbox.find('ymin').text)),
            round(float(bndbox.find('xmax').text)),
            round(float(bndbox.find('ymax').text))
        ])
    return boxes

def write_xml(boxes, original_file, new_file):
    tree = etree.parse(original_file)
    root = tree.getroot()
    for i, member in enumerate(root.findall('object')):
        bndbox = member.find('bndbox')
        bndbox.find('xmin').text = str(boxes[i][0])
        bndbox.find('ymin').text = str(boxes[i][1])
        bndbox.find('xmax').text = str(boxes[i][2])
        bndbox.find('ymax').text = str(boxes[i][3])
        root.find('filename').text = os.path.basename(new_file).replace('.xml', '.png')
    tree.write(new_file)

def augment_image(image_path, xml_path, save_dir, prefix, iteration):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = read_xml(xml_path)

    unique_seed = hash((os.path.basename(image_path) + str(iteration))) & 0xffffffff

    random.seed(unique_seed)
    np.random.seed(unique_seed)    

    # augmentation
    transform = A.Compose([
        # apply horizontal flip with 50% probability
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),  # Adjust brightness and contrast
        A.Rotate(limit=30, p=0.5),  # Rotate within a range of -20 to 20 degrees with a 50% probability
        A.RandomResizedCrop(height=320, width=320, scale=(0.9, 1.0), p=0.5),
        A.PadIfNeeded(min_height=320, min_width=320, border_mode=cv2.BORDER_REFLECT)  # Pad if needed after crop to maintain size 320x320


    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))

    transformed = transform(image=image, bboxes=boxes)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)

    img_save_path = os.path.join(save_dir, 'images', f"{prefix}_{os.path.basename(image_path)}")
    xml_save_path = os.path.join(save_dir, 'annotations', f"{prefix}_{os.path.basename(xml_path)}")

    cv2.imwrite(img_save_path, transformed_image)
    write_xml(transformed_bboxes, xml_path, xml_save_path)

def main():
    # TODO: Set the paths, image_dir and xml_dir are original images and annotations, save_dir is the directory to save augmented images and annotations
    image_dir = 'compressed/micro_all_frames_training/images'
    xml_dir = 'compressed/micro_all_frames_training/Annotations'
    save_dir = 'augmented'

    os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'annotations'), exist_ok=True)

    images = os.listdir(image_dir)
    for img_file in images:
        image_path = os.path.join(image_dir, img_file)
        xml_file = img_file.replace('.png', '.xml')
        xml_path = os.path.join(xml_dir, xml_file)

        # create two augmented images for each image, change the range to create more
        for i in range(4): 
            augment_image(image_path, xml_path, save_dir, f"aug_{i}", i)  

if __name__ == '__main__':
    main()
