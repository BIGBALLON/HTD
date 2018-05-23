from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.regularizers import l2

class lenet(object):
    def __init__(self):
        pass 
        
    def build(self,img_input,classes_num):
        weight_decay = 0.0001
        x = Conv2D(6, (5, 5), padding='valid', activation = 'relu', 
            kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(img_input)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = Conv2D(16, (5, 5), padding='valid', activation = 'relu', 
            kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(120, activation = 'relu', 
            kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
        x = Dense(84, activation = 'relu', 
            kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
        x = Dense(classes_num, activation = 'softmax', 
            kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
        return x
