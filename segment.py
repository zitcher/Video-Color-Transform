import torchvision
from PIL import Image
import torch
import numpy as np
import cv2 as cv

def segment(image):
    with torch.no_grad():
        image = image/255
        im = torch.tensor(image).unsqueeze(0).permute(0, 3, 1, 2).float()
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        model = model.eval()
        out = model(im)[0]

        combined_mask = np.zeros((image.shape[0], image.shape[1]))

        for i in range(len(out['masks'])):
            mask = out['masks'].squeeze(1)[i].numpy()
            combined_mask = np.maximum(combined_mask, mask)

        combined_mask = 255*(combined_mask - np.min(combined_mask)) / np.max(combined_mask)

        return combined_mask.astype(np.uint8)

if __name__ == '__main__':
    im = np.array(Image.open("./data/abby.jpg"))
    print(im.shape)
    seg = segment(im)
    print(seg.shape)

    seg = Image.fromarray(seg)
    seg.save('./data/out.png')