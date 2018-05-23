from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D
from keras.regularizers import l2

class wresnet(object):
    def __init__(self):
        pass 

    def conv3x3(self, x,filters):
        return Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1), padding='same',
        kernel_initializer="he_normal",
        kernel_regularizer=l2(0.0005),
        use_bias=False)(x)

    def residual_block(self, x,out_filters,increase=False):
        stride = (1,1)
        if increase:
            stride = (2,2)
        o1     = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
        conv_1 = Conv2D(out_filters,kernel_size=(3,3),strides=stride,padding='same',kernel_initializer="he_normal",kernel_regularizer=l2(0.0005),use_bias=False)(o1)
        o2     = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1))
        conv_2 = Conv2D(out_filters, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer="he_normal",kernel_regularizer=l2(0.0005),use_bias=False)(o2)
        if increase or self.in_filters != out_filters:
            proj  = Conv2D(out_filters,kernel_size=(1,1),strides=stride,padding='same',kernel_initializer="he_normal",kernel_regularizer=l2(0.0005),use_bias=False)(o1)
            block = add([conv_2, proj])
        else:
            block = add([conv_2, x])
        return block

    def wide_residual_layer(self, x,out_filters,increase=False):
        x = self.residual_block(x,out_filters,increase)
        self.in_filters = out_filters
        for _ in range(1,int(self.n_stack)):
            x = self.residual_block(x,out_filters)
        return x

    def build(self, img_input,classes_num,depth=28,k=10):
        print('Wide-Resnet %dx%d' %(depth, k))
        self.n_filters  = [16, 16*k, 32*k, 64*k]
        self.n_stack    = (depth - 4) / 6
        self.in_filters = 16

        x = self.conv3x3(img_input,self.n_filters[0])
        x = self.wide_residual_layer(x,self.n_filters[1])
        x = self.wide_residual_layer(x,self.n_filters[2],increase=True)
        x = self.wide_residual_layer(x,self.n_filters[3],increase=True)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = Activation('relu')(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(classes_num,activation='softmax',kernel_initializer="he_normal",kernel_regularizer=l2(0.0005),use_bias=False)(x)
        return x
