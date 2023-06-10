import requests
import cv2
import numpy as np
import torch

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
        file_name = f"video_{message_id}.mp4"
        
        with open(file_name, "wb") as file:
            file.write(response.content)
        
        print(f"Video saved as {file_name}")
    else:
        print("Failed to download video")

def process_video(message_id):
    input_video_path = f"video_{message_id}.mp4"
    video = cv2.VideoCapture(input_video_path)

    # 書き出し先の指定
    # Video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    # Output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec (e.g., 'XVID', 'MJPG', 'mp4v')
    output_video = cv2.VideoWriter(f'output_{message_id}.mp4', fourcc, fps, (width, height))

    while video.isOpened():
        ret, frame = video.read()

            # フレームの読み込みに問題があればループを終了
        if not ret:
            break

        cfg, pose_estimator = get_predictor()

        # ここにフレームに対する処理を記述する
        outputs = pose_estimator(frame)

        n_boxes = len(outputs["instances"].pred_boxes)
        outputs["instances"].pred_boxes = torch.empty((n_boxes,4)) #replace bounding box with an empty tensor to prevent the Visualizer from drawing the bounding box
        v = Visualizer(frame[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out_frame = v.get_image()[:, :, ::-1]
        # cv2.imwrite(f"output_{message_id}.jpg",out_frame)
        output_video.write(out_frame)
        
    # Release resources
    video.release()
    output_video.release()
    cv2.destroyAllWindows()