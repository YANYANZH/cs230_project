"""
    Write each feature from predict.mrc into single mrc files for display in Chimera.
    run python3 postprocess.py predict.mrc
    Written by Yanyan Zhao.
    """

import mrcfile as mrc
import sys
from utils import tiff_to_np
import numpy as np

image = sys.argv[1]
with mrc.open(image) as f:
    img = f.data
    new_mask = np.float32(tiff_to_np(img))

for i in range(6):
    with mrc.new(image.split('.')[0] + str('_') + str(i) +'.mrc') as nf:
        nf.set_data(new_mask[i, :, :])








