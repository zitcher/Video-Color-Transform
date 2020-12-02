import pytube
import os

def download_video(vidid):
    url = f'https://youtube.com/watch?v={vidid}'
    youtube = pytube.YouTube(url)

    # download correct
    youtube.streams.filter(res="720p").first().download('./', youtube.title)

if __name__ == "__main__":
    download_video('WwfVaehcdfE')