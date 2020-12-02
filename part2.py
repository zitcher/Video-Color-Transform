import argparse, os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.linalg import inv, norm
from scipy.linalg import sqrtm

SOURCE_PATH = './data'
OUTPUT_PATH = './output'

def read_image(path):
    '''
    read_image(path) takes a string which is the name of the input image (which should be stored in ./data), then read it with
    Lab color space (0..225 for 8-bit).
    '''
    img = cv2.imread(os.path.join(SOURCE_PATH, path)) # BGR uint8
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab) # Lab uint8
    return img

def write_image(img, path): # Input Lab uint8
    '''
    write_image(img, path) takes a 3-channel image in Lab color space (0..255 for 8-bit), and shift it back to BGR space then store it
    with given name (path). The output would be stored in ./output directory.
    '''
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

def write_video(frames, path):
    return

def get_cdf(lum, testing=False): # If testing then plot the cdf
    '''
    Given a single channel luminance image. We are able to establish corresponding cumulative density function (like cumulative histogram). If the
    testing mode in on, then we will plot the function. This option is by default off.
    '''
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
    '''
    This function counts the frequency (number of occurrence time) of eacb value possible in L channel (in Lab space). If the cumulative flag is on, then
    we returns cumulative histogram (unnormalized cdf), else we return unnormalized pdf.
    '''
    hist = np.zeros(256, dtype=np.uint)
    for i in range(hist.shape[0]):
        hist[i] = np.count_nonzero(lum <= i) if cumulative else np.count_nonzero(lum == i)
    return hist

def luminance_transfer(input_frame, target_frame):
    '''
    This part could be done through calculating the generalized inverse of cdf. (Topics in Optimal Transportation, by Villani 2003; Ch 2.2.1)
    '''
    # Convolving frames with a boundary preserving Gaussian kernel of std=.1 (optional but stated in the paper)
    input_temp, target_temp = input_frame[:,:,0], target_frame[:,:,0]
    input_temp, target_temp = cv2.bilateralFilter(input_temp, 5, .1, .1), cv2.bilateralFilter(target_temp, 5, .1, .1)
    # Calculate the corresponding cdf for each frame
    input_cdf, target_cdf = get_cdf(input_temp), get_cdf(target_temp)
    # Initialize the generalized inverse of cdf (here we know the domain is discrete)
    input2target = np.zeros(input_cdf.shape[0], dtype=np.uint)
    # Establish the inverse function
    for i in range(input_cdf.shape[0]):
        input2target[i] = np.min(np.argwhere(target_cdf >= input_cdf[i]))
    # Perform the inverse transfer
    output_frame = input_frame.copy()
    output_frame[:,:,0] = input2target[input_frame[:,:,0]]
    return output_frame

def get_band_feature(frame, band_indices=None, alpha_mask=None):
    '''
    Given a specific frame and numpy array of indices in the band, we calculate the mean and covariance in the two chrominance channels.
    If there is no band indices inputs, then it would calculate global features.
    WARNING: the input indices are ravel indices!
    '''
    chro = np.reshape(frame[:,:,1:], (frame.shape[0]*frame.shape[1],2))
    if band_indices is not None:
        # Perform linear masking for calculating band mean/covariance
        chro = chro[band_indices]
        mu, sig = np.sum(chro * np.stack([alpha_mask, alpha_mask], axis=-1), axis=0) / np.sum(alpha_mask), np.cov(chro.transpose(), aweights=alpha_mask)
        return mu, sig
    # If indices are not given, then treat the whole frame as one band
    mu, sig = np.mean(chro, axis=0), np.cov(chro.transpose())
    return mu, sig

def get_frame_feature(frame, split_bands=True):
    '''
    The function returns a list of mean and covariance matrix for each band (list of pairs), as well as the indices for each band and corresponding masks which
    covers the each band.
    The bands intersection parts are used for smoothing the color transfer, which is a linear alpha masking
    '''
    # Split the three bands with different levels of luminance
    lum = frame[:,:,0].flatten() # Ravel view
    indices = np.argsort(lum, axis=None)
    size = indices.shape[0]
    # Split to three bands or no split depending on parameter
    slices = [indices[:int(.4*size)], indices[int(.3*size):int(.7*size)], indices[int(.6*size):]] if split_bands else [indices]
    masks, alpha_masks = [], []
    # Calculate the intersection between bands
    intersection_size = int(.1*size)
    for i, band in enumerate(slices):
        # Calculate mask for blending the transfered image, and alpha mask for calculating the mean/covariance
        if i == 0:
            # No left intersection
            mask = get_band_mask(band, frame.shape[:2], right_falloff_intersection=intersection_size)
            alpha_mask = get_alpha_mask(band.shape, right_falloff_intersection=intersection_size)
        elif i == len(slices)-1:
            # No right intersection
            mask = get_band_mask(band, frame.shape[:2], left_falloff_intersection=intersection_size)
            alpha_mask = get_alpha_mask(band.shape, left_falloff_intersection=intersection_size)
        else:
            mask = get_band_mask(band, frame.shape[:2], left_falloff_intersection=intersection_size, right_falloff_intersection=intersection_size)
            alpha_mask = get_alpha_mask(band.shape, left_falloff_intersection=intersection_size, right_falloff_intersection=intersection_size)
        masks.append(mask)
        alpha_masks.append(alpha_mask)
    # Get the features (mean and covariance) for each band
    feats = [get_band_feature(frame, band, mask) for band, mask in zip(slices, alpha_masks)]
    return feats, slices, masks

# TODO: Compare PCA method and Linear Monge-Kantorovitch method. (Maybe consider Cholesky method.)
def get_T(mu_input, sig_input, mu_target, sig_target):
    '''
    The default transfer matrix T which is calculated through linear Monge-Kantorovitch method.
    '''
    sqrt_sig_input = sqrtm(sig_input)
    T = sqrtm(sqrt_sig_input @ sig_target @ sqrt_sig_input)
    inv_sqrt_sig_input = inv(sqrt_sig_input)
    T = inv_sqrt_sig_input @ T @ inv_sqrt_sig_input
    return T

def get_alpha_mask(shape, left_falloff_intersection=0, right_falloff_intersection=0):
    '''
    Get the alpha mask with given left/right intersect range and shape
    '''
    mask = np.ones(shape)
    if left_falloff_intersection != 0:
        mask[:left_falloff_intersection] = np.arange(0., 1., 1./left_falloff_intersection)
    if right_falloff_intersection != 0:
        mask[-right_falloff_intersection:] = np.arange(1., 0., -1./right_falloff_intersection)
    return mask

def get_band_mask(indices, shape, left_falloff_intersection=0, right_falloff_intersection=0):
    '''
    Create a band mask where all pixels with given ravel indices are masked as ones, while others are kept to be zeros.
    '''
    mask = np.zeros(shape).flatten()
    # Mask the given indices to be ones, others to zeros
    mask[indices] = 1
    # Smooth the intersection part
    if left_falloff_intersection != 0:
        mask[indices[:left_falloff_intersection]] = np.arange(0., 1., 1./left_falloff_intersection)
    if right_falloff_intersection != 0:
        mask[indices[-right_falloff_intersection:]] = np.arange(1., 0., -1./right_falloff_intersection)
    # Reshape back to 2d-image
    mask = np.reshape(mask, shape)
    return np.stack([mask, mask], axis=-1)

def chrominance_transfer(input_frame, target_frame, multi_band=True):
    '''
    The function takes input frame and target frame, and keep the luminance unchanged. It tries to get the frame features, and calculate
    the transfer matrix based on the features (mean/covariance) for each band. 
    '''
    output_frame = np.zeros(input_frame.shape, dtype=np.uint8)
    output_frame[:,:,0] = input_frame[:,:,0]
    input_feats, input_slices, slice_masks = get_frame_feature(input_frame, multi_band)
    target_feats, _, _ = get_frame_feature(target_frame, multi_band)
    for i, ((mu_input, sig_input), (mu_target, sig_target), slice_indices, mask) in enumerate(zip(input_feats, target_feats, input_slices, slice_masks)):
        T = get_T(mu_input, sig_input, mu_target, sig_target)
        output_frame[:,:,1:] += (((input_frame[:,:,1:] - mu_input) @ T + mu_target) * mask).astype(np.uint8)
        # write_image(output_frame, 'sample_'+str(i)+'.jpg')
    return output_frame

def get_distance(feats1, feats2):
    '''
    Inputs are lists of pairs: each pair has a mean vector and a covariance matrix. 
    '''
    d = [np.trace(cov1+cov2)-2*sqrtm(sqrtm(cov1)@cov2@sqrtm(cov1))+norm(mu1-mu2) for (mu1, cov1), (mu2, cov2) in zip(feats1, feats2)]
    return sum(d)

# TODO: Have not touched yet
def find_representative_frame(frames):
    pass

def main():
    '''
    Simple tests.
    '''
    target = read_image('tree.jpg')
    img = read_image('example.jpg')
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
    # TODO: Add parsers and arguments
    #parser = argparse.ArgumentParser(description="CSCI1290 - Final Project")
    #parser.add_argument("method", )
