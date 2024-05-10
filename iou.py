import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from edge_impulse_linux.image import ImageImpulseRunner
from PIL import Image, ImageDraw


def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    wi, hi = max(0, xi2 - xi1), max(0, yi2 - yi1)
    inter = wi * hi
    union = w1 * h1 + w2 * h2 - inter
    return inter / union


def parse_voc_annotation(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xtl = int(float(bndbox.find('xmin').text))
        ytl = int(float(bndbox.find('ymin').text))
        xbr = int(float(bndbox.find('xmax').text))
        ybr = int(float(bndbox.find('ymax').text))
        objects.append((xtl, ytl, xbr, ybr))
    return objects


def calculate_new_bbox(ground_truth, original_height, target_size, the_image):
    xtl, ytl, xbr, ybr = ground_truth
    bbox_width = xbr - xtl
    bbox_height = ybr - ytl

    width_diff = 640 - 480  # 640 is the original width of the image
    xtl = xtl - width_diff / 2
    xbr = xbr - width_diff / 2

    resize_ratio = target_size / original_height
    xtl = xtl * resize_ratio
    ytl = ytl * resize_ratio
    bbox_width = bbox_width * resize_ratio
    bbox_height = bbox_height * resize_ratio

    # = Image.fromarray(the_image)
    # draw = ImageDraw.Draw(image)
    # draw.rectangle([xtl, ytl, xtl + bbox_width, ytl + bbox_height], outline="red", width=2)
    # image.save(os.path.join(output_dir, "new_" + img_filename))

    return [(xtl, ytl, bbox_width, bbox_height)]


dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dir_path, "rpi-yolo-final-320-1-linux-x86_64-v3.eim")
image_dir = os.path.join(dir_path, "images")

annotation_dir = os.path.join(dir_path, "annotations")
# output_dir = os.path.join(dir_path, "new_testing_images")
# os.makedirs(output_dir, exist_ok=True)

runner = ImageImpulseRunner(model_path)
model_info = runner.init()
print("Model initialized:", model_info)

precisions = []


def resize_image_fit_shortest_axis(img):
    h, w = img.shape[:2]
    if w > h:
        diff = w - h
        # crop half of the difference from the left and right
        image = img[:, diff // 2:w - diff // 2]
    else:
        diff = h - w
        # crop half of the difference from the top and bottom
        image = img[diff // 2:h - diff // 2]
    image = cv2.resize(image, (320, 320))
    return image


for img_filename in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_filename)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    original_img = img.copy()  # Make a copy for drawing and saving
    img = resize_image_fit_shortest_axis(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    features, cropped = runner.get_features_from_image(img)
    res = runner.classify(features)

    annotation_path = os.path.join(annotation_dir, os.path.splitext(img_filename)[0] + ".xml")
    ground_truths = parse_voc_annotation(annotation_path)

    new_ground_truths = calculate_new_bbox(ground_truths[0], 480, 320, img)

    matches = []
    if "bounding_boxes" in res["result"]:
        for bb in res["result"]["bounding_boxes"]:
            pred_box = (bb['x'], bb['y'], bb['width'], bb['height'])
            for gt_box in new_ground_truths:
                iou = calculate_iou(pred_box, (gt_box[0], gt_box[1], gt_box[2], gt_box[3]))
                if iou >= 0.5:
                    print("Matched", pred_box, gt_box, iou)
                    matches.append(1)
                    break
            else:
                matches.append(0)
                print("No match", pred_box)
    if len(matches) > 0:
        precision = sum(matches) / len(matches)
    else:
        precision = 0
    precisions.append(precision)
average_precision = sum(precisions) / len(precisions) if precisions else 0
print("Average Precision at IoU >= 0.5:", average_precision)

runner.stop()