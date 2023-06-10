from flask import Flask, jsonify, request, send_from_directory
import urllib.request
import subprocess
import requests
from dotenv import load_dotenv
import os
from utils import *

app = Flask(__name__)

load_dotenv()
CHANNEL_ACCESS_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CURRENT_URL = os.getenv("CURRENT_URL")

@app.route('/', methods=['GET', 'POST'])
def webhook():
    if request.method == 'POST':
        data = request.get_json()
        replyToken = data['events'][0]['replyToken']
        
        if data['events'][0]['message']['type'] == 'video':
            messageId = data['events'][0]['message']['id']
            download_video(messageId, CHANNEL_ACCESS_TOKEN)
            process_video(messageId)
            convert_images_to_video(messageId,f"videos/output_{messageId}.mp4", 30)
            delete_directory_contents(f"frames/{messageId}")
            print("video created")
            url, headers, data = create_video_response(messageId, replyToken, CHANNEL_ACCESS_TOKEN, CURRENT_URL)
            response = requests.post(url, headers=headers, json=data)
            
            # delete frames and original video
            # os.remove(f"download/video_{messageId}.mp4")
            # os.remove(f"images/output_{messageId}.jpg")
            # os.remove(f"videos/output_{messageId}.mp4")
            # urllib.image_dir, output_video_path, fpsrequest.urlretrieve(url_link, 'video_name.mp4') 
            #response = {'message': 'Webhook received'}

            return response #jsonify(response)
        else:
            url, headers, data = create_text_response(replyToken, CHANNEL_ACCESS_TOKEN)
            response = requests.post(url, headers=headers, json=data)
            
            # data = request.get_json()
            # print(data)
            # response = {'message': 'Webhook received'}
            return response
    elif request.method == 'GET':
        return 'Hello World'


# @app.route('/api/data')
# def get_data():
#     data = {'message': 'This is sample data'}
#     return jsonify(data)

@app.route('/images/<image_id>')
def serve_image(image_id):
    # Determine the filename based on the image_id
    filename = f"output_{image_id}.jpg"
    return send_from_directory('images', filename)

@app.route('/videos/<video_id>')
def serve_video(video_id):
    # Determine the filename based on the video_id
    filename = f"output_{video_id}.mp4"
    return send_from_directory('videos', filename)


if __name__ == '__main__':
    app.run(debug=True)
