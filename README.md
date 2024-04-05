# How to use scripts
- Install necessary packages.
- Run the frame_extractor.py script to extract the frames from the video. The script can be run from the terminal with arguments:
  - ``python3 frame_extractor.py video_name video_path annotation_path``, where video_name is the name of the video file, video_path is the path to the video file, and annotation_path is the path to the annotation file.
- An example:
  - ```python3 frame_extractor.py 'GX0110088' 'videos/GX0110088_no_audio.MP4' 'annotations/annotations.xml'```

- This will create the frames directory with the appropriate nested folder frames_video_name, and the extracted frames will be saved in this folder.
- Run the image_cropper script to crop out the objects from the frames. The script can be run from the terminal with arguments:
  - ``python3 image_cropper.py target_image_size video_name annotation_path``, where target_image_size is the size of the image to be cropped(edge of the square), video_name is the name of the video file, and annotation_path is the path to the annotation file.
- An example:
  - ```python3 image_cropper.py 244 'GX0110088' 'annotations/annotations.xml'```

Note that video_name parameter is just to create a folder with appropriate video name for easier navigation. 
Make sure you have enough space on your disk to store the frames.