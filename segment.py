import torchvision
from PIL import Image
import torch
import numpy as np
import cv2
from tqdm import tqdm, trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def segment(images):
    with torch.no_grad():
        normalized = []
        for image in images:
            image = image/255
            normalized.append(torch.tensor(image).permute(2, 0, 1).float().to(device))

        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)
        model = model.eval()
        outputs = model(normalized)

        masks = []

        for i in range(len(outputs)):
            out = outputs[i]
            combined_mask = np.zeros((images[i].shape[0], images[i].shape[1]))
            for i in range(len(out['masks'])):
                mask = out['masks'].squeeze(1)[i].cpu().numpy()
                combined_mask = np.maximum(combined_mask, mask)

            combined_mask = 255*(combined_mask - np.min(combined_mask)) / np.max(combined_mask)
            masks.append(combined_mask.astype(np.uint8))

        return masks 

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

def segment_video(frames):
    out_frames = []
    window_sz = 8
    for i in trange(0, len(frames), window_sz):
        sub_section = frames[i:i+window_sz]
        out_frames = out_frames + segment(sub_section)
        
    return out_frames

if __name__ == '__main__':
    # im = np.array(Image.open("./data/abby.jpg"))
    # im2 = np.array(Image.open("./data/horseback.jpg"))
    # print(im.shape)
    # segs = segment([im, im2])

    # for i in range(len(segs)):
    #     seg = segs[i]
    #     print(seg.shape)
    #     seg = Image.fromarray(seg)
    #     seg.save('./data/seg_{}.png'.format(i))

    frames = load_video('./data/all_results/src_models/amelie.mp4')
    print(frames[0].shape)
    segs = segment_video(frames)

    fps = 30
    video_filename = './data/out.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(video_filename, fourcc, fps, (segs[0].shape[1], segs[0].shape[0]))
    print(len(segs))
    for i in range(len(segs)):
        print(segs[i].shape)
        seg = Image.fromarray(segs[i])
        seg.save('./data/vid/seg_{}.png'.format(i))
        out.write(np.stack([segs[i], segs[i], segs[i]], axis=2))

    