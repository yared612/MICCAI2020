import tensorflow.keras.layers as KL
from tensorflow.keras.layers import Conv2D ,Activation ,Add ,concatenate ,BatchNormalization, \
MaxPooling2D ,ZeroPadding2D ,Dropout ,Flatten ,Dense ,AveragePooling2D, Conv2DTranspose, add, \
UpSampling2D ,Lambda ,multiply
from tensorflow.keras import Model
from tensorflow.keras import backend as K
import warnings
warnings.filterwarnings("ignore")

def identity_block(input_tensor, kernel_size, filters, stage, block,use_bias=True):
    
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',use_bias=use_bias,kernel_initializer = 'he_normal')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',name=conv_name_base + '2b', use_bias=use_bias,kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',use_bias=use_bias,kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block,strides=(2, 2), use_bias=True):
    
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', use_bias=use_bias,kernel_initializer = 'he_normal')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',name=conv_name_base + '2b', use_bias=use_bias,kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias,kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1', use_bias=use_bias,kernel_initializer = 'he_normal')(input_tensor)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x
def expend_as(tensor, rep):
     return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                          arguments={'repnum': rep})(tensor)

def attention_block(x, gating, inter_shape, name):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)
    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  
    shape_theta_x = K.int_shape(theta_x)

    phi_g = Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(1,1),
                                 padding='same')(phi_g)  

    concat_xg = add([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(2,2),name=name+'_weight')(sigmoid_xg)  

    upsample_psi = expend_as(upsample_psi, shape_x[3])


    y = multiply([upsample_psi, x])
    y = AveragePooling2D((2,2))(y)

    result = Conv2D(shape_g[3], (1, 1), padding='same')(y)
    result_bn = BatchNormalization()(result)
    return result_bn

def resnet(input_shape ,output_dim):
    input_layer = KL.Input(shape=input_shape, name='data')
    s1 = ZeroPadding2D((3, 3))(input_layer)
    
    s1 = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True, padding='valid',kernel_initializer = 'he_normal')(s1)
    s1 = BatchNormalization()(s1)
    s1 = Activation('relu')(s1)
    
    # Stage 2
    s2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(s1)
    s2 = conv_block(s2, 3, [64, 64, 256], stage=2, block='a',strides=(1,1))
    s2 = identity_block(s2, 3, [64, 64, 256], stage=2, block='b')
    s2 = identity_block(s2, 3, [64, 64, 256], stage=2, block='c')
    s2 = attention_block(s1,s2,256,'att_s1_s2')
    # Stage 3
    s3 = conv_block(s2, 3, [128, 128, 512], stage=3, block='a')
    s3 = identity_block(s3, 3, [128, 128, 512], stage=3, block='b')
    s3 = identity_block(s3, 3, [128, 128, 512], stage=3, block='c')
    s3 = identity_block(s3, 3, [128, 128, 512], stage=3, block='d')
    s3 = attention_block(s2,s3,512,'att_s2_s3')
    # Stage 4
    s4 = conv_block(s3, 3, [256, 256, 1024], stage=4, block='a')
    s4 = identity_block(s4, 3, [256, 256, 1024], stage=4, block='b')
    s4 = identity_block(s4, 3, [256, 256, 1024], stage=4, block='c')
    s4 = identity_block(s4, 3, [256, 256, 1024], stage=4, block='d')
    s4 = identity_block(s4, 3, [256, 256, 1024], stage=4, block='e')
    s4 = identity_block(s4, 3, [256, 256, 1024], stage=4, block='f')
    s4 = attention_block(s3,s4,512,'att_s3_s4')
    # Stage 5
    s5 = conv_block(s4, 3, [512, 512, 2048], stage=5, block='a')
    s5 = identity_block(s5, 3, [512, 512, 2048], stage=5, block='b')
    s5 = identity_block(s5, 3, [512, 512, 2048], stage=5, block='c')
    s5 = attention_block(s4,s5,512,'att_s4_s5')
    
    block_end = AveragePooling2D(pool_size=(7, 7), padding='valid')(s5)
    block_end = Flatten()(block_end)
    block_end = Dropout(0.4)(block_end)
    block_end = Dense(1024, activation='relu',kernel_initializer = 'he_normal')(block_end)
    block_end = Dropout(0.4)(block_end)
    block_end = Dense(output_dim, activation='sigmoid',kernel_initializer = 'he_normal')(block_end)
    model = Model(inputs=input_layer, outputs=block_end)
    return model

#model = resnet((512,512,3),2)
#model.summary(line_length=120)