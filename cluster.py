from sklearn_extra.cluster import KMedoids
from segment import load_video
import numpy as np
from scipy import spatial
from tqdm import trange
import cv2
from part2 import get_frame_feature
from scipy.linalg import fractional_matrix_power


def raw_dist(frame1, frame2):
    mean1, cov1 = get_band_feature(frame1)
    mean2, cov2 = get_band_feature(frame2)
    
    tr = np.trace(cov1 + cov2 - 2 * fractional_matrix_power((fractional_matrix_power(cov1, 0.5) @ cov2 @ fractional_matrix_power(cov1, 0.5)), 0.5))
    md = np.linalg.norm(mean1 - mean2)**2

    return tr + md

# L2 optimal transport distance images
'''
The paper references Ferredan 2012 OPTIMAL TRANSPORT MIXING OF GAUSSIAN TEXTURE MODELS 
which has the 2*(...) term inside the trace, so I went with that. Otherwise this 
returns a square matrix and isn't usable as a distance function.

'''
def dist(stats1, stats2):
    dist = 0
    for i in range(3):
        mean1 = stats1[i:i+2]
        cov1 = np.reshape(stats1[i+2:i+6], (2, 2))

        mean2 = stats2[i:i+2]
        cov2 = np.reshape(stats2[i+2:i+6], (2, 2))
        
        tr = np.trace(cov1 + cov2 - 2 * fractional_matrix_power((fractional_matrix_power(cov1, 0.5) @ cov2 @ fractional_matrix_power(cov1, 0.5)), 0.5))
        md = np.linalg.norm(mean1 - mean2)**2

        dist += tr + md

    return dist


def get_stats(frame):
    feats, slices, masks = get_frame_feature(frame)

    stack = []
    for mean, cov in feats:
        stack.append(mean)
        stack.append(np.reshape(cov, -1))

    stack = np.concatenate(stack, axis=0)
    return stack
'''
Expects LAB image format
'''
def find_video_kmediods(video_frames):
    return KMedoids(n_clusters=len(video_frames)//30, metric=dist).fit(video_frames), video_frames.shape[1:]


def load_lab_video(path):
    lab_frame_stats = []
    lab_frames = []
    frames = load_video(path)
    print("getting frame bands")
    for frame in frames:
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2Lab)
        lab_frames.append(lab)
        lab_frame_stats.append(get_stats(lab))
    return lab_frame_stats, lab_frames

def video_kmediod(lab_frames):
    print("clustering")
    medoids, m_shape = find_video_kmediods(np.array(lab_frames))
    centers = medoids.cluster_centers_
    indices = medoids.medoid_indices_
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
                medoids_over_30.append((centers[label], indices[label]))

    return medoids_over_30


if __name__ == '__main__':
    vp = './data/all_results/src_models/amelie.mp4'
    lab_frame_stats, lab_frames = load_lab_video(vp)
    print(video_kmediod(lab_frame_stats))

    # frames = load_video(vp)

    # im1 = cv2.cvtColor(frames[10], cv2.COLOR_RGB2Lab)
    # im2 = cv2.cvtColor(frames[200], cv2.COLOR_RGB2Lab)

    # print(dist(im1, im2))
