from sklearn_extra.cluster import KMedoids
from segment import load_video
import numpy as np
from scipy import spatial
from tqdm import trange
import cv2
from part1 import get_band_feature
from scipy.linalg import fractional_matrix_power

# L2 optimal transport distance images
'''
The paper references Ferredan 2012 OPTIMAL TRANSPORT MIXING OF GAUSSIAN TEXTURE MODELS 
which has the 2*(...) term inside the trace, so I went with that. Otherwise this 
returns a square matrix and isn't usable as a distance function.

'''
def dist(frame1, frame2):
    # TODO: Switch to splitting into bands and summing
    mean1, cov1 = get_band_feature(frame1)
    mean2, cov2 = get_band_feature(frame2)
    
    tr = np.trace(cov1 + cov2 - 2 * fractional_matrix_power((fractional_matrix_power(cov1, 0.5) @ cov2 @ fractional_matrix_power(cov1, 0.5)), 0.5))
    md = np.linalg.norm(mean1 - mean2)**2

    return tr + md


'''
Expects LAB image format
'''
def find_video_kmediods(video_frames):
    return KMedoids(n_clusters=video_frames//30, metric=dist).cluster_centers_

def find_and_load_video_kmediod(path):
    lab_frames = []
    frames = load_video(path)
    for frame in frames:
        lab_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2Lab))
    
    medoids = find_video_kmediods(lab_frames)
    centers = medoids.cluster_centers_
    labels = medoids.labels_

    # only take medoids w/ 30 or more elements
    medoids_over_30 = []
    counts = dict()
    for label in labels:
        if label not in counts:
            counts[label] = 1
        elif counts[label] >= 30:
            continue
        else:
            counts[label] += 1
            if counts[label] >= 30:
                over_30.append(centers[label])

    return medoids_over_30


if __name__ == '__main__':
    vp = './data/all_results/src_models/amelie.mp4'
    # print(find_and_load_video_kmediod(vp))

    frames = load_video(vp)

    im1 = cv2.cvtColor(frames[10], cv2.COLOR_RGB2Lab)
    im2 = cv2.cvtColor(frames[200], cv2.COLOR_RGB2Lab)

    print(dist(im1, im2))
