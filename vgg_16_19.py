import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

def get_session(gpu_fraction=0.3):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session(1.0))

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
try:
    import h5py
except ImportError:
    h5py = None

img_input = (256, 256, 2)

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=img_input))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="tf"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="tf"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="tf"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="tf"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="tf"))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

def VGG_19(weights_path=None):
    model = Sequential()
    # model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(ZeroPadding2D((1,1),input_shape=img_input))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="tf"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="tf"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="tf"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="tf"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="tf"))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path, by_name=True)

    return model

if __name__ == "__main__":


    # im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
    # im[:,:,0] -= 103.939
    # im[:,:,1] -= 116.779
    # im[:,:,2] -= 123.68
    # im = im.transpose((2,0,1))
    # im = np.expand_dims(im, axis=0)

    batch_size = 16
    num_classes = 3
    epochs = 100

    # write a function to select x_train y_train x_test y_test
    data_x = np.load('image_data_256x256.npy')
    data_y = np.load('labels.npy')

    # concatenate together and shuffle the data
    data_x = np.swapaxes(np.swapaxes(data_x,1,2),2,3)
    data_y = np.array([data_y]).T

    # shuffle the data first
    from sklearn.utils import shuffle
    data_x, data_y = shuffle(data_x, data_y)

    # Test pretrained model
    # model = VGG_19("vgg19_weights_th_dim_ordering_th_kernels.h5")
    model = VGG_16("vgg16_weights_tf_dim_ordering_tf_kernels.h5")
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    model.summary()

    from sklearn.cross_validation import train_test_split
    from sklearn.model_selection import KFold
    import keras
    cross_validation = True
    data_augmentation = True
    batch_size = 10
    epochs = 5000
    test_set_size = 0.2
    val_set_size = 0.2
    cv_split_size = 10
    processed_data = []
    if cross_validation:
        all_index = [_ for _ in range(len(data_x))]
        # cross-validataion data set
        kf = KFold(n_splits=cv_split_size)

        for train, test in kf.split(all_index):
            print("Train size: {}, Test size: {}".format(train.shape, test.shape))
            x_train = data_x[train]
            y_train = data_y[train]
            x_test = data_x[test]
            y_test = data_y[test]

            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_set_size)
            # Convert class vectors to binary class matrices.
            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_val = keras.utils.to_categorical(y_val, num_classes)
            y_test = keras.utils.to_categorical(y_test, num_classes)

            x_train = x_train.astype('float32')
            x_val = x_val.astype('float32')
            x_test = x_test.astype('float32')
            x_train /= 255
            x_val /= 255
            x_test /= 255

            processed_data.append([(x_train,y_train),(x_val, y_val),(x_test, y_test)])
        print "Cross validation data has been split!"

    predict = []
    for _ in range(len(processed_data)):
        (x_test, y_test) = processed_data[_][2]
        for i in range(len(x_test)):
            # input_x = x_test[i]
            out = model.predict(x_test)
            predict.append(out)
        # print np.argmax(out)

    # out = model.predict(im)
    # print np.argmax(out)
