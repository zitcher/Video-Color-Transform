import sklearn
from segment import load_video
import numpy as np
from scipy import spatial
from tqdm import trange

'''
Expects LAB image format
'''
def find_video_mediod(video_frames):
    data = []
    for frame in video_frames:
        countsa, optionsg = np.histogram(frame[:,:,1], bins=np.arange(256))
        countsb, optionsb = np.histogram(frame[:,:,2], bins=np.arange(256))

        # get color histogram as a vector
        hist = np.concatenate((countsr, countsg, countsb), axis=0)
        data.append(hist)

    data = np.array(data)

    # compute all pairwise distances
    pairdist = spatial.distance.cdist(data, data)

    # mediod defined as element with the least distance to all others in the cluster
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
    # separate into sections of size 30
    for i in trange(0, len(frames), window_sz):
        sub_section = frames[i:i+window_sz]

        # find the mediod of each section
        representatives.append(i + find_video_mediod(sub_section))
        
    return representatives




def find_and_load_video_kmediod(path):
    lab_frames = []
    frames = load_video(path)
    for frame in frames:
        lab_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2Lab))
    
    find_video_kmediod(lab_frames)


if __name__ == '__main__':
    vp = './data/all_results/src_models/amelie.mp4'
    print(find_and_load_video_kmediod(vp))