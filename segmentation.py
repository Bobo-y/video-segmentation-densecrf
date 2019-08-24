import argparse
import os
import json
from PIL import Image
import numpy as np
import scipy.misc as misc
import warnings
from network import Seg
from utils import *


warnings.filterwarnings('ignore')


def _main_(args):
    config_path = args.conf
    input_path = args.input
    output_path = args.output

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpu']
    infer_model = Seg().seg_network()
    infer_model.load_weights(config['train']['saved_weights'])

    image_paths = []

    if os.path.isdir(input_path):
        for inp_file in os.listdir(input_path):
            image_paths += [input_path + inp_file]
    else:
        image_paths += [input_path]

    image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]

    # the main loop
    for image_path in image_paths:
        image = Image.open(image_path).resize((512, 512))
        image = normalize(np.expand_dims(np.array(image), 0))
        res = infer_model.predict(image, batch_size=1)
        res = np.squeeze(res, axis=0)
        res = sigmoid(res)
        res_np = res.astype(np.float32)
        cond = np.greater_equal(res_np, 0.5).astype(np.int)
        misc.imsave(os.path.join(output_path, image_path.split('/')[-1].split('.')[0] + '.png'), cond[:, :, 0])


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Predict with a trained model')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--input', help='path to an image, a directory of images, a video, or webcam')
    argparser.add_argument('-o', '--output', default='output/', help='path to output directory')
    args = argparser.parse_args()
    _main_(args)
