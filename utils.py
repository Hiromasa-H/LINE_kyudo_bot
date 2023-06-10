import requests
import cv2
import numpy as np
import torch
import os

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def get_predictor():
    print("loading model...")
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    print("model loaded")
    return cfg, predictor

def download_video(message_id, channel_access_token):
    url = f"https://api-data.line.me/v2/bot/message/{message_id}/content"
    headers = {"Authorization": f"Bearer {channel_access_token}"}
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        # Assuming the video file has a .mp4 extension, you can adjust it as needed
        file_name = f"download/video_{message_id}.mp4"
        
        with open(file_name, "wb") as file:
            file.write(response.content)
        
        print(f"Video saved as {file_name}")
    else:
        print("Failed to download video")

def process_video(message_id):
    input_video_path = f"download/video_{message_id}.mp4"
    video = cv2.VideoCapture(input_video_path)

    # output video does not work for some reason
    # Video properties
    # width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    # Output video
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec (e.g., 'XVID', 'MJPG', 'mp4v')
    # output_video = cv2.VideoWriter(f'output_{message_id}.mp4', fourcc, fps, (width, height))
    
    if not os.path.exists(f"frames/{message_id}"):
        os.makedirs(f"frames/{message_id}")


    cfg, pose_estimator = get_predictor()

    # video = cv2.VideoCapture(input_video_path)
    frame_count = 0

    while video.isOpened():
        ret, frame = video.read()

            # フレームの読み込みに問題があればループを終了
        if not ret:
            break

        # ここにフレームに対する処理を記述する
        outputs = pose_estimator(frame)

        n_boxes = len(outputs["instances"].pred_boxes)
        outputs["instances"].pred_boxes = torch.empty((n_boxes,4)) #replace bounding box with an empty tensor to prevent the Visualizer from drawing the bounding box
        v = Visualizer(frame[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out_frame = v.get_image()[:, :, ::-1]
        cv2.imwrite(f"frames/{message_id}/output_{frame_count:04d}.jpg",out_frame)
        out_frame = cv2.imread(f"frames/{message_id}/output_{frame_count:04d}.jpg")
        # output_video.write(out_frame)

        frame_count += 1
        
    # Release resources
    video.release()
    # output_video.release()
    cv2.destroyAllWindows()
    # print("video created")

def convert_images_to_video(messageId, output_video_path, fps):

    image_dir = f"frames/{messageId}"
    # Get the list of image files in the directory
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])

    # sort the files numerically
    image_files = sorted(image_files, key=lambda item: (int(item.partition('.')[0]) if item[0].isdigit() else float('inf'), item))

    # Read the first image to get the dimensions
    first_image_path = os.path.join(image_dir, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec (e.g., 'XVID', 'MJPG', 'mp4v')
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Loop over the image files and write each frame to the video
    for i, image_file in enumerate(image_files):
        # print(f'Processing {image_file}')
        image_path = os.path.join(image_dir, image_file)
        frame = cv2.imread(image_path)
        output_video.write(frame)
        if i == 0:
            cv2.imwrite(f"images/output_{messageId}.jpg", frame)

    # Release the video writer
    output_video.release()

def delete_directory_contents(directory):
    # Iterate over all files and subdirectories in the given directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Construct the absolute file path
            file_path = os.path.join(root, file)

            # Delete the file
            os.remove(file_path)

        for dir in dirs:
            # Construct the absolute directory path
            dir_path = os.path.join(root, dir)

            # Recursively delete the subdirectory and its contents
            delete_directory_contents(dir_path)
            os.rmdir(dir_path)

        os.rmdir(directory)

def create_video_response(message_id, response_id, channel_access_token,current_url):
    
    url = 'https://api.line.me/v2/bot/message/reply'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {channel_access_token}'
    }
    data = {
        'replyToken': response_id,
        'messages': [
            {
             "type": "video",
            "originalContentUrl": f"{current_url}/videos/{message_id}",
            "previewImageUrl": f"{current_url}/images/{message_id}"
            }
        ]
    }

    return url, headers, data

def create_text_response(response_id, channel_access_token):
    
    url = 'https://api.line.me/v2/bot/message/reply'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {channel_access_token}'
    }
    data = {
        'replyToken': response_id,
        'messages': [
            {
                "type": "text",
                "text": "申し訳ございません、動画以外のファイルには対応しておりません。"
            }
        ]
    }

    return url, headers, data
