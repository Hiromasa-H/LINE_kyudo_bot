import requests

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
    return