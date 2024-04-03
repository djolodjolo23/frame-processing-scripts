import cv2
import xml.etree.ElementTree as ET
import os

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    frame_numbers = []

    for track in root.findall('track'):
        if track.attrib['label'] == 'queen':
            for box in track.findall('./box'):
                frame_numbers.append(int(box.attrib['frame']))

    return frame_numbers

if not os.path.exists('frames'):
    os.makedirs('frames')

video_path = 'GX011088_no_audio.MP4'
xml_path = 'annotations.xml'

frame_numbers = parse_xml(xml_path)


cap = cv2.VideoCapture(video_path)

current_frame = 0;

while True:
    success, frame = cap.read()
    if not success:
        break

    if current_frame in frame_numbers:

        frame_path = f'frames/frame_{current_frame}.jpg'
        cv2.imwrite(frame_path, frame)

    current_frame += 1

    if current_frame == 360: # stopping point for now
        break

cap.release()



