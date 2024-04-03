import cv2
import xml.etree.ElementTree as ET
import os


def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    frame_nums = []

    for track in root.findall('track'):
        if track.attrib['label'] == 'queen':
            for box in track.findall('./box'):
                frame_nums.append(int(box.attrib['frame']))

    return frame_nums


if not os.path.exists('frames'):
    os.makedirs('frames')

if not os.path.exists('frames/frames_GX011088'):
    os.makedirs('frames/frames_GX011088')

video_path = 'GX011088_no_audio.MP4' # some problems with cv2 and mp4 files when there is audio, so I made a copy of the video without audio
xml_path = 'annotations.xml'

frame_numbers = parse_xml(xml_path)

cap = cv2.VideoCapture(video_path)

current_frame = 0

while True:
    success, frame = cap.read()
    if not success:
        break
    if current_frame in frame_numbers:
        frame_path = f'frames/frames_GX011088/frame_{current_frame}.jpg'
        cv2.imwrite(frame_path, frame)
    current_frame += 1

    if current_frame == 200:  # stopping point for now
        break

cap.release()


