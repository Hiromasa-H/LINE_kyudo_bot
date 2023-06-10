from flask import Flask, jsonify, request
import urllib.request
import subprocess
import requests
from dotenv import load_dotenv
import os
from utils import download_video

app = Flask(__name__)

load_dotenv()
CHANNEL_ACCESS_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")


@app.route('/', methods=['GET', 'POST'])
def webhook():
    if request.method == 'POST':
        try:
            data = request.get_json()
            messageId = data['events'][0]['message']['id']
            download_video(messageId, CHANNEL_ACCESS_TOKEN)
            # urllib.request.urlretrieve(url_link, 'video_name.mp4') 
            response = {'message': 'Webhook received'}
            return jsonify(response)
        except Exception as e:
            data = request.get_json()
            print(data)
            response = {'message': 'Webhook received'}
            return jsonify(response)
    elif request.method == 'GET':
        return 'Hello World'


@app.route('/api/data')
def get_data():
    data = {'message': 'This is sample data'}
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
