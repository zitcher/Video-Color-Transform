import sklearn
from segment import load_video
import numpy as np
from scipy import spatial
from tqdm import trange

def find_video_mediod(video_frames):
    data = []
    for frame in video_frames:
        countsr, optionsr = np.histogram(frame[:,:,0], bins=np.arange(256))
        countsg, optionsg = np.histogram(frame[:,:,1], bins=np.arange(256))
        countsb, optionsb = np.histogram(frame[:,:,2], bins=np.arange(256))

        hist = np.concatenate((countsr, countsg, countsb), axis=0)
        data.append(hist)

    data = np.array(data)
    pairdist = spatial.distance.cdist(data, data)

    best = np.argmin(np.sum(pairdist, axis=0))
    return best

'''
The paper states:
In practice, we run the K-medoids algorithm on the frames of the model video 
to produce one representative model frame for every 30 frames of the model 
video, i.e., one for every second of a model video sampled at 30fps.

I believe what this means is that we should break the video into 30 frame segments
and find a mediod for each segment.
'''
def find_video_kmediod(frames):
    representatives = []
    window_sz = 30
    for i in trange(0, len(frames), window_sz):
        sub_section = frames[i:i+window_sz]
        representatives.append(i + find_video_mediod(sub_section))
        
    return representatives

if __name__ == '__main__':
    vp = './data/all_results/src_models/amelie.mp4'
    frames = load_video(vp)
    print(find_video_kmediod(frames))