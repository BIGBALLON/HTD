from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, add, Activation, GlobalAveragePooling2D
from keras import regularizers

class resnet(object):
    def __init__(self):
        pass
    
    def residual_block(self, intput,out_channel,increase=False):
        if increase:
            stride = (2,2)
        else:
            stride = (1,1)

        pre_bn   = BatchNormalization(momentum=0.9, epsilon=1e-5)(intput)
        pre_relu = Activation('relu')(pre_bn)

        conv_1 = Conv2D(out_channel,kernel_size=(3,3),strides=stride,padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(0.0001),
                        use_bias=False)(pre_relu)
        bn_1   = BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1)
        relu1  = Activation('relu')(bn_1)
        conv_2 = Conv2D(out_channel,kernel_size=(3,3),strides=(1,1),padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(0.0001),
                        use_bias=False)(relu1)
        if increase:
            projection = Conv2D(out_channel,
                                kernel_size=(1,1),
                                strides=(2,2),
                                padding='same',
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(0.0001),
                                use_bias=False)(pre_relu)
            block = add([conv_2, projection])
        else:
            block = add([intput,conv_2])
        return block

    def build(self, img_input,classes_num=10, stack_n=5):
        # build model
        # total layers = stack_n * 3 * 2 + 2
        # stack_n = 5 by default, total layers = 32
        # input: 32x32x3 output: 32x32x16
        x = Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),padding='same',
                   kernel_initializer="he_normal",
                   kernel_regularizer=regularizers.l2(0.0001),
                   use_bias=False)(img_input)

        # input: 32x32x16 output: 32x32x16
        for _ in range(stack_n):
            x = self.residual_block(x,16,False)

        # input: 32x32x16 output: 16x16x32
        x = self.residual_block(x,32,True)
        for _ in range(1,stack_n):
            x = self.residual_block(x,32,False)
        
        # input: 16x16x32 output: 8x8x64
        x = self.residual_block(x,64,True)
        for _ in range(1,stack_n):
            x = self.residual_block(x,64,False)

        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = Activation('relu')(x)
        x = GlobalAveragePooling2D()(x)

        # input: 64 output: 10
        x = Dense(classes_num,activation='softmax',
                  kernel_initializer="he_normal",
                  kernel_regularizer=regularizers.l2(0.0001),
                  use_bias=False)(x)
        return x
