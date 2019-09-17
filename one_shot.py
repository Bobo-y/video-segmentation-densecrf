from network import Seg
from utils import *
from keras.optimizers import Adam
import argparse
import os
from PIL import Image
import json
import numpy as np
from scipy import misc


def _main_(args):
    config_path = args.conf
    input_path = args.input
    output_path = args.output
    image = args.image
    mask = args.mask

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpu']
    fine_tune_num = config['train']['fine_tune']

    net = Seg().seg_network()
    net.load_weights(config['train']['saved_weights'])
    opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    net.compile(opt, loss=loss)

    image = Image.open(image).resize((512, 512))
    image = normalize(np.expand_dims(np.array(image), 0))
    mask = Image.open(mask).resize((512, 512))
    mask = normalize(np.expand_dims(np.expand_dims(np.array(mask), 0), 3))
    # fine-tune
    for i in range(fine_tune_num):
        fine_tune_loss = net.train_on_batch(image, mask)
        print(i, fine_tune_loss)

    image_paths = []
    for inp_file in os.listdir(input_path):
        image_paths += [input_path + inp_file]

    image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]
    for image_path in image_paths:
        image = Image.open(image_path).resize((512, 512))
        image = normalize(np.expand_dims(np.array(image), 0))
        res = net.predict(image, batch_size=1)
        res = np.squeeze(res, axis=0)
        res = sigmoid(res)
        res_np = res.astype(np.float32)
        cond = np.greater_equal(res_np, 0.5).astype(np.int)
        misc.imsave(os.path.join(output_path, image_path.split('/')[-1].split('.')[0] + '.png'), cond[:, :, 0])


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='one shot segmentation with a parent trained model')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--image', help='path of first frame')
    argparser.add_argument('-m', '--mask', help='path of mask for first frame')
    argparser.add_argument('-d', '--input', help='path of images directory to seg')
    argparser.add_argument('-o', '--output', default='output/', help='path to output directory')
    args = argparser.parse_args()
    _main_(args)
