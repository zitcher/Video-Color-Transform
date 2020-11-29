import argparse, os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.linalg import inv

SOURCE_PATH = './data'
OUTPUT_PATH = './output'

def read_image(path):
    img = cv2.imread(os.path.join(SOURCE_PATH, path)) # BGR uint8
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab) # Lab uint8
    return img

def write_image(img, path): # Input Lab uint8
    img = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)
    cv2.imwrite(os.path.join(OUTPUT_PATH, path), img)
    return

# TODO: Not finished (Not sured about if we want to get all frames at once)
def read_video(path):
    cap = cv2.videoCapture(path)
    while cap.isOpened():
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab) # Not sure if they use BGR or RGB for video
    cap.release()
    return

def get_cdf(lum, testing=False): # If testing then plot the cdf
    hist = get_histogram(lum).astype(np.float64)
    hist /= hist[-1]
    if testing:
        plt.plot(np.arange(256), hist, 's-', color='r', label='cdf')
        plt.xlabel('value')
        plt.ylabel('probability')
        plt.legend()
        plt.show()
    return hist

def get_histogram(lum, cumulative=True):
    hist = np.zeros(256, dtype=np.uint)
    for i in range(hist.shape[0]):
        hist[i] = np.count_nonzero(lum <= i) if cumulative else np.count_nonzero(lum == i)
    return hist

# TODO: Convolving frames with a boundary preserving Gaussian kernel of std=.1
def luminance_transfer(input_frame, target_frame):
    input_cdf, target_cdf = get_cdf(input_frame[:,:,0]), get_cdf(target_frame[:,:,0])
    input2target = np.zeros(input_cdf.shape[0], dtype=np.uint)
    for i in range(input_cdf.shape[0]):
        input2target[i] = np.min(np.argwhere(target_cdf >= input_cdf[i]))
    input_frame[:,:,0] = input2target[input_frame[:,:,0]]
    return input_frame

def get_band_feature(frame, band_indices=None):
    if not band_indices:
        chro = np.reshape(frame[:,:,1:], (frame.shape[0]*frame.shape[1],2))
        mu, sig = np.mean(chro, axis=0), np.cov(chro.transpose())
        return mu, sig
    # TODO: for band specific
    return None

# TODO: cubic falloff alpha blending for band intersection (smooth the transformation)
def get_frame_feature(frame):
    # Split the three bands with different levels of luminance.
    lum = frame[:,:,0].flatten() # Flattened
    indices = np.argsort(lum, axis=None)
    size = indices.shape[0]
    slices = [indices[:int(.4*size)], indices[int(.3*size):int(.7*size)], indices[int(.6*size):]]
    # Get the features (mean and covariance) for each band
    # feats = [get_band_feature(frame, band) for band in slices]
    # TODO: For testing global transfer
    feats = get_band_feature(frame)
    return feats#, slices

def chrominance_transfer(input_frame, target_frame):
    input_feats, target_feats = get_frame_feature(input_frame), get_frame_feature(target_frame)
    # TODO: Perform the transfer based on the features
    mu_input, sig_input = input_feats
    mu_target, sig_target = target_feats
    sqrt_sig_input = sig_input#np.sqrt(sig_input)
    T = np.sqrt(sqrt_sig_input @ sig_target @ sqrt_sig_input)
    inv_sqrt_sig_input = inv(sqrt_sig_input)
    T = inv_sqrt_sig_input @ T @ inv_sqrt_sig_input
    input_frame[:,:,1:] = (input_frame[:,:,1:] - mu_input) @ T.transpose() + mu_target
    return input_frame

# TODO: Segmentation
def get_background_segment(frame):
    pass

# TODO: Have not touched yet
def find_representative_frame(frames):
    pass

def main():
    target = read_image('example.jpg')
    img = read_image('tree.jpg')
    plt.plot(np.arange(256), get_cdf(img[:,:,0]), 's-', color='g', label='original', markersize=1)
    output = luminance_transfer(img, target)
    plt.plot(np.arange(256), get_cdf(target[:,:,0]), 's-', color='r', label='target', markersize=1)
    plt.plot(np.arange(256), get_cdf(output[:,:,0]), 's-', color='b', label='output', markersize=1)
    plt.xlabel('value')
    plt.ylabel('probability')
    plt.legend()
    plt.show()
    output = chrominance_transfer(output, target)
    write_image(output, 'sample.jpg')
    return

if __name__ == '__main__':
    main()
    #parser = argparse.ArgumentParser(description="CSCI1290 - Final Project")
    #parser.add_argument("method", )
