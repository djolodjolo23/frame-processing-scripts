from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import os

predefined_square_size = 400  #testing with 400x400
additional_padding = 10
video_name = 'GX011088'
folder_path = 'frames'
initial_xml = 'annotations.xml'


cropped_folder_path = f'cropped/{video_name}'
os.makedirs(cropped_folder_path, exist_ok=True)

tree = ET.parse(initial_xml)
root = tree.getroot()

for track in root.findall('.//track'):
    if track.attrib.get('label') == 'queen':
        for box in track.findall('box'):
            frame_num = box.attrib['frame']

            image_path = os.path.join(folder_path, f'frames_{video_name}/frame_{frame_num}.jpg')
            if not os.path.exists(image_path):
                continue

            original_image = Image.open(image_path)
            xtl, ytl, xbr, ybr = [float(box.attrib[attr]) for attr in ['xtl', 'ytl', 'xbr', 'ybr']]

            bbox_width = xbr - xtl
            bbox_height = ybr - ytl

            initial_square_size = max(bbox_width, bbox_height)

            predefined_square_size_temp = predefined_square_size
            if initial_square_size > predefined_square_size:
                predefined_square_size = round(initial_square_size) + additional_padding # adding predefined padding pixels to the square size to make sure the object is not cropped out

            padding_height = (predefined_square_size - bbox_height) / 2
            padding_width = (predefined_square_size - bbox_width) / 2

            # the new square crop area values
            crop_left = xtl - padding_width
            crop_upper = ytl - padding_height
            crop_right = xbr + padding_width
            crop_lower = ybr + padding_height

            cropped_area = original_image.crop((int(crop_left), int(crop_upper), int(crop_right), int(crop_lower)))
            new_image = Image.new('RGB', (predefined_square_size, predefined_square_size), (255, 255, 255))

            # pasting the cropped image to the new image
            paste_x = (predefined_square_size - cropped_area.width) // 2
            paste_y = (predefined_square_size - cropped_area.height) // 2
            new_image.paste(cropped_area, (paste_x, paste_y))

            if predefined_square_size_temp != predefined_square_size:
                new_image = new_image.resize((predefined_square_size_temp, predefined_square_size_temp))
                resize_ratio = predefined_square_size_temp / predefined_square_size
                new_tlx = padding_width * resize_ratio
                new_tly = padding_height * resize_ratio
                new_brx = (padding_width * resize_ratio) + (bbox_width * resize_ratio)
                new_bry = (padding_height * resize_ratio) + (bbox_height * resize_ratio)
            else:
                new_tlx = padding_width
                new_tly = padding_height
                new_brx = padding_width + bbox_width
                new_bry = padding_height + bbox_height

            box.set('xtl', str(new_tlx))
            box.set('ytl', str(new_tly))
            box.set('xbr', str(new_brx))
            box.set('ybr', str(new_bry))

            draw = ImageDraw.Draw(new_image)
            draw.rectangle([(new_tlx, new_tly), (new_brx, new_bry)], outline='red', width=3)

            new_image.save(f'cropped/{video_name}/cropped_frame{frame_num}.jpg')
            predefined_square_size = predefined_square_size_temp

new_xml_path = f'annotations_updated_{video_name}.xml'
tree.write(new_xml_path)
