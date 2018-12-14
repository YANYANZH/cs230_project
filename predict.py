"""
Using trained model to predict features in 512 by 512 tomogram image. 
python3 predict.py Image/image.tiff label/label.tiff
Wirtten by Yanyan Zhao
"""

import torch
import torch.nn as nn
import model
import sys
from PIL import Image
import mrcfile as mrc
import numpy as np


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = model.Net(1, 6)
    net = nn.DataParallel(net)

    cp_dir = 'checkpoints/cp.pth'
    net.load_state_dict(torch.load(cp_dir))
    net.to(device)
    net.eval()
    img = sys.argv[1]
    img, h ,w = predict(net,img)

    tiff_to_mrc(img)
    label = sys.argv[2]
    label = np.array(Image.open(label))
    acc = np.sum(img == label)
    acc = acc/float(h*w)
    print(acc)

def predict(net,input):

    img = Image.open(input)
    img = np.array(img)
    h, w = img.shape
    img = img.reshape((1,1,h,w))
    img = torch.from_numpy(img)

    with torch.no_grad():
        output = net(img)
        output = torch.argmax(output, dim=1)
        output = np.array(output.cpu())
        output = output.astype(np.float32).reshape((h,w))
    return output, h, w

def tiff_to_mrc(input):
    with mrc.new('predict.mrc') as f:
        f.set_data(input)


if __name__ == '__main__':
    main()
