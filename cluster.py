from sklearn_extra.cluster import KMedoids
#from segment import load_video
import numpy as np
from scipy import spatial
from tqdm import trange
import cv2
from part2 import get_frame_feature, get_band_feature, chrominance_transfer, get_distance
from scipy.linalg import fractional_matrix_power, sqrtm

'''
def raw_dist(frame1, frame2):
    mean1, cov1 = get_band_feature(frame1)
    mean2, cov2 = get_band_feature(frame2)
    
    tr = np.trace(cov1 + cov2 - 2 * sqrtm(sqrtm(cov1) @ cov2 @ sqrtm(cov1)))
    md = np.linalg.norm(mean1 - mean2)**2

    return tr + md
'''

# L2 optimal transport distance images
'''
The paper references Ferredan 2012 OPTIMAL TRANSPORT MIXING OF GAUSSIAN TEXTURE MODELS 
which has the 2*(...) term inside the trace, so I went with that. Otherwise this 
returns a square matrix and isn't usable as a distance function.

'''

def load_video(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            cap.release()
            cv2.destroyAllWindows()
            return frames
        frames.append(frame)

def dist(stats1, stats2):
    dist = 0
    for i in range(0, 18, 6):
        mean1 = stats1[i:i+2]
        cov1 = np.reshape(stats1[i+2:i+6], (2, 2))

        mean2 = stats2[i:i+2]
        cov2 = np.reshape(stats2[i+2:i+6], (2, 2))
        
        tr = np.trace(cov1 + cov2 - 2 * sqrtm(sqrtm(cov1) @ cov2 @ sqrtm(cov1)))
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

def video_transfer(source, target):
    lab_frame_stats, lab_frames = load_lab_video(target)

    lab_frames_source_stats, lab_frame_source = load_lab_video(source)
    mediods = video_kmediod(lab_frames_source_stats)

    output = []
    for i in range(len(lab_frame_stats)):
        best_dist = None
        best_match = None
        for center, index in mediods:
            frame_dst = dist(center, lab_frame_stats[i])
            if best_dist == None:
                best_match = lab_frame_source[index]
                best_dist = frame_dst
                continue

            if best_dist > frame_dst:
                best_match = lab_frame_source[index]
                best_dist = frame_dst
        output.append(chrominance_transfer(lab_frames[i], best_match))
    
    return output

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

def naive_transfer(vp, target):
    target_frames = load_video(target)
    source_frames = load_video(vp)
    target_frames, source_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2Lab) / 255. for frame in target_frames], [cv2.cvtColor(frame, cv2.COLOR_RGB2Lab) / 255. for frame in source_frames]
    source_feats = [(np.mean(frame, axis=(0,1)), np.std(frame, axis=(0,1))) for frame in source_frames]
    target_feats = [(np.mean(frame, axis=(0,1)), np.std(frame, axis=(0,1))) for frame in target_frames]
    output = []
    for i, frame in enumerate(source_frames):
        i_target = i / (len(source_frames) - 1) * (len(target_frames) - 1)
        i_target = min(int(i_target), len(target_frames) - 1)
        mu_source, std_source = source_feats[i]
        mu_target, std_target = target_feats[i_target]
        out_frame = frame.copy()
        out_frame[:,:,1:] = (out_frame[:,:,1:] - mu_source[1:]) / std_source[1:]
        out_frame[:,:,1:] = out_frame[:,:,1:] * std_target[1:] + mu_target[1:]
        out_frame[:,:,1:] = np.clip(out_frame[:,:,1:], 0, 1)
        out_frame *= 255.
        out_frame = out_frame.astype(np.uint8)
        output.append(out_frame)
    return output

if __name__ == '__main__':
    vp = 'source.mp4'
    target = 'target.mp4'
    
    output_naive = naive_transfer(vp, target)
    for frame in output_naive:
        frame = cv2.cvtColor(frame, cv2.COLOR_Lab2RGB)
        cv2.imshow('Frame', frame)
        cv2.waitKey(100)
    result = None
    size = output_naive[0].shape[:2]
    size = size[::-1]
    result = cv2.VideoWriter('naive.mp4', cv2.VideoWriter_fourcc('m','p','4','v'), 30, size)
    for frame in output_naive:
        frame = cv2.cvtColor(frame, cv2.COLOR_Lab2RGB)
        result.write(frame)
    result.release()
    
    output = video_transfer(vp, target)
    for frame in output:
        frame = cv2.cvtColor(frame, cv2.COLOR_Lab2RGB)
        cv2.imshow('Frame', frame)
        cv2.waitKey(100)
    result = None
    size = output[0].shape[:2]
    size = size[::-1]
    result = cv2.VideoWriter('result.mp4', cv2.VideoWriter_fourcc('m','p','4','v'), 30, size)
    for frame in output:
        frame = cv2.cvtColor(frame, cv2.COLOR_Lab2RGB)
        result.write(frame)
    result.release()
    
    diff = np.stack([np.sum((frame1-frame2)**2, axis=None) for frame1, frame2 in zip(output_naive, output)])
    im1, im2 = output_naive[np.argmax(diff)], output[np.argmax(diff)]
    im1, im2 = cv2.cvtColor(im1, cv2.COLOR_Lab2BGR), cv2.cvtColor(im2, cv2.COLOR_Lab2BGR)
    
    
    #print(video_kmediod(lab_frame_stats))

    # frames = load_video(vp)

    # im1 = cv2.cvtColor(frames[10], cv2.COLOR_RGB2Lab)
    # im2 = cv2.cvtColor(frames[200], cv2.COLOR_RGB2Lab)

    # print(dist(im1, im2))
