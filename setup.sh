touch .env
echo "LINE_CHANNEL_SECRET='<your token here>' \n CURRENT_URL = '<deployed URL here>' " >> .env
pip3 install -r requirements.txt
python -m pip3 install 'git+https://github.com/facebookresearch/detectron2.git'