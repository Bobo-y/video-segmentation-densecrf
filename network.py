# coding=utf-8
from keras import Input, Model
from keras.applications.vgg16 import VGG16
from keras.layers import Concatenate, Conv2D, UpSampling2D, BatchNormalization


class Seg:

    def __init__(self):
        self.input_img = Input(name='input_img',
                               shape=(512, 512, 3),
                               dtype='float32')
        self.out_channel = 1
        vgg16 = VGG16(input_tensor=self.input_img, 
                      weights='imagenet',
                      include_top=False)
        self.locked_layers = False
        if self.locked_layers:
            # locked first two conv layers
            locked_layers = [vgg16.get_layer('block1_conv1'),
                             vgg16.get_layer('block1_conv2')]
            for layer in locked_layers:
                layer.trainable = False
        self.vgg_pools = [vgg16.get_layer('block%d_pool' % i).output
                          for i in range(1, 6)]

    def seg_network(self):

        def decoder(layer_input, skip_input, channel, last_block=False):
            if not last_block:
                concat = Concatenate(axis=-1)([layer_input, skip_input])
                bn1 = BatchNormalization()(concat)
            else:
                bn1 = BatchNormalization()(layer_input)
            conv_1 = Conv2D(channel, 1,
                            activation='relu', padding='same')(bn1)
            bn2 = BatchNormalization()(conv_1)
            conv_2 = Conv2D(channel, 3,
                            activation='relu', padding='same')(bn2)
            return conv_2

        d1 = decoder(UpSampling2D((2, 2))(self.vgg_pools[4]), self.vgg_pools[3], 128)
        d2 = decoder(UpSampling2D((2, 2))(d1), self.vgg_pools[2], 64)
        d3 = decoder(UpSampling2D((2, 2))(d2), self.vgg_pools[1], 32)
        d4 = decoder(UpSampling2D((2, 2))(d3), self.vgg_pools[0], 32)
        d5 = decoder(UpSampling2D((2, 2))(d4), None, 32, True)

        mask = Conv2D(self.out_channel, 3, padding='same')(d5)
        model = Model(inputs=self.input_img, outputs=[mask])
        from keras.optimizers import Adam
        opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(opt, loss='mse')
        model.summary()
        return model


Seg().seg_network()