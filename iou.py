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
    xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    wi, hi = max(0, xi2 - xi1), max(0, yi2 - yi1)
    inter = wi * hi
    union = w1*h1 + w2*h2 - inter
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

# Additional functionality to calculate precision at multiple IoU thresholds
def calculate_precision_at_thresholds(matches_dict, ground_truths, predictions, thresholds, copied_img):
    for iou_threshold in thresholds:
        matches = matches_dict.setdefault(iou_threshold, [])
        if predictions:
            for pred_box in predictions:
                matched = False
                for gt_box in ground_truths:
                    iou = calculate_iou(pred_box, (gt_box[0], gt_box[1], gt_box[2], gt_box[3]))
                    if iou >= iou_threshold:
                        matched = True
                        # save the image with both bounding boxes
                        if (iou <= iou_threshold + 0.05):
                            recalculated_bboxes = [reverse_bbox_transform(gt_box, 480, 640, 320)]
                            recalculated_predictions = [reverse_bbox_transform(pred_box, 480, 640, 320)]
                            draw_both_bboxes(copied_img, recalculated_bboxes, recalculated_predictions, iou)
                            #draw_both_bboxes(copied_img, ground_truths, predictions, iou)
                        break
                matches.append(int(matched))
        else:
            matches.extend([0] * len(ground_truths))  # No predictions mean zero matches

def calculate_map(precisions):
    return {threshold: sum(matches) / len(matches) if matches else 0 for threshold, matches in precisions.items()}

def calculate_new_bbox(ground_truth, original_height, original_width, target_size):
    if (original_width != original_height):
        xtl, ytl, xbr, ybr = ground_truth
        bbox_width = xbr - xtl
        bbox_height = ybr - ytl

        width_diff = original_width-original_height # 640 is the original width of the image
        xtl = xtl - width_diff / 2
        xbr = xbr - width_diff / 2
        resize_ratio = target_size / original_height
        xtl = xtl * resize_ratio
        ytl = ytl * resize_ratio
        bbox_width = bbox_width * resize_ratio
        bbox_height = bbox_height * resize_ratio
        # = Image.fromarray(the_image)
        #draw = ImageDraw.Draw(image)
        #draw.rectangle([xtl, ytl, xtl + bbox_width, ytl + bbox_height], outline="red", width=2)
        #image.save(os.path.join(output_dir, "new_" + img_filename))

        return [(xtl, ytl, bbox_width, bbox_height)]
    else:
        bbox_width = ground_truth[2] - ground_truth[0]
        bbox_height = ground_truth[3] - ground_truth[1]
        return [(ground_truth[0], ground_truth[1], bbox_width, bbox_height)]

def reverse_bbox_transform(resized_bbox, original_height, original_width, target_size):
    # Extract values from the resized bounding box
    xtl, ytl, resized_width, resized_height = resized_bbox

    # Step 1: Reverse the scaling applied during the resize
    resize_ratio = target_size / original_height
    xtl /= resize_ratio
    ytl /= resize_ratio
    original_bbox_width = resized_width / resize_ratio
    original_bbox_height = resized_height / resize_ratio

    # Step 2: Reverse the width difference adjustment (center alignment)
    width_diff = original_width - original_height
    xtl += width_diff / 2

    # Create the original bounding box coordinates
    xbr = xtl + original_bbox_width
    ybr = ytl + original_bbox_height

    return (xtl, ytl, original_bbox_width, original_bbox_height)




def draw_both_bboxes(image, ground_truths, predicted_boxes, iou):
    the_image = Image.fromarray(image)
    draw = ImageDraw.Draw(the_image)
    for box in ground_truths:
        draw.rectangle([box[0], box[1], box[0] + box[2], box[1] + box[3]], outline="red", width=2)
    for box in predicted_boxes:
        draw.rectangle([box[0], box[1], box[0] + box[2], box[1] + box[3]], outline="green", width=2)
    the_image.save(os.path.join(output_dir, f"{iou}_" + img_filename))

def resize_image_fit_shortest_axis(img):
    h, w = img.shape[:2]
    if w > h:
        diff = w - h
        image = img[:, diff//2:w-diff//2]
    else:
        diff = h - w
        image = img[diff//2:h-diff//2]
    image = cv2.resize(image, (320, 320))
    return image


dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dir_path, "models/yolo/rpi-yolo-final-320-3-linux-x86_64-v1.eim")
image_dir = os.path.join(dir_path, "rpi_all_frames_3/testing/images")

annotation_dir = os.path.join(dir_path, "rpi_all_frames_3/testing/annotations")
output_dir = os.path.join(dir_path, "MaP_examples_original_size")
os.makedirs(output_dir, exist_ok=True)

runner = ImageImpulseRunner(model_path)
model_info = runner.init()
print("Model initialized:", model_info)

# Main processing
iou_thresholds = np.arange(0.5, 1, 0.05)
iou_thresholds = np.round(iou_thresholds, 2)  # Round to two decimal places
# Ensuring that 0.95 is included in the thresholds
if 0.95 not in iou_thresholds:
    iou_thresholds = np.append(iou_thresholds, 0.95)
precisions = {threshold: [] for threshold in iou_thresholds}

average_precisions = []

for img_filename in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_filename)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    copied_img = img.copy()
    img = resize_image_fit_shortest_axis(img)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    features, cropped = runner.get_features_from_image(img)
    res = runner.classify(features)

    annotation_path = os.path.join(annotation_dir, os.path.splitext(img_filename)[0] + ".xml")
    ground_truths = parse_voc_annotation(annotation_path)
    new_ground_truths = calculate_new_bbox(ground_truths[0], 480, 640, 320)

    predicted_boxes = []
    if "bounding_boxes" in res["result"]:
        predicted_boxes = [(bb['x'], bb['y'], bb['width'], bb['height']) for bb in res["result"]["bounding_boxes"]]

    calculate_precision_at_thresholds(precisions, new_ground_truths, predicted_boxes, iou_thresholds, copied_img)

average_precisions = calculate_map(precisions)
map_value = sum(average_precisions.values()) / len(average_precisions)
print("Average Precisions:", average_precisions)
print("Mean Average Precision (MaP):", map_value)

runner.stop()