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

KTF.set_session(get_session(0.5))

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np

import os
num_classes = 3
image_shape = (256,256,2)
# num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_neuron_classification_trained_model.h5'

# write a function to select x_train y_train x_test y_test
data_x = np.load('image_data_256x256.npy')
data_y = np.load('labels.npy')

# concatenate together and shuffle the data
data_x = np.swapaxes(np.swapaxes(data_x,1,2),2,3)
data_y = np.array([data_y]).T

# shuffle the data first
from sklearn.utils import shuffle
data_x, data_y = shuffle(data_x, data_y)

def build_model(img_shape = (256, 256, 2)):
    # define the model
    # CNN structure
    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding='same',
                     input_shape=img_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model

model = build_model(img_shape = image_shape)

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.00005, decay=1e-5)
# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#from keras.optimizers import SGD
#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

cross_validation = True
data_augmentation = True

from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
test_set_size = 0.1
val_set_size = 0.3
cv_split_size = 5

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
else:
    # to set the train:validation:test = 0.6:0.2:0.2
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=test_set_size)
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
    print "Data has been split!"

cross_validation = True
data_augmentation = True
batch_size = 8
epochs = 1000

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

# trainging and test process
def learn_and_test(x_train, y_train,
                  x_val, y_val,
                  x_test, y_test):
    if not data_augmentation:
        print('Not using data augmentation.')
        for _ in range(epochs):
            model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=1,
                  validation_data=(x_val, x_val),
                  shuffle=False,
                  callbacks=[early_stopping])
            print "Epoch {}/{}:".format(_ + 1, epochs)
        (loss,accur) = model.evaluate(x=x_test, y=y_test, batch_size=batch_size)
        print "Model accuary after training: {}".format(accur)
        return loss, accur
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        for _ in range(epochs):
            model.fit_generator(datagen.flow(x_train, y_train,
                                             batch_size=batch_size),
                                steps_per_epoch=x_train.shape[0] // batch_size,
                                epochs=1,
                                validation_data=(x_val, y_val),
                                workers=4,
                                callbacks=[early_stopping])
            print "Epoch {}/{}:".format(_ + 1, epochs)
        (loss,accur) = model.evaluate(x=x_test, y=y_test, batch_size=batch_size)
        print "Model accuary after training: {}".format(accur)
        return loss, accur


if not cross_validation:
    (x_train,y_train),(x_val, y_val),(x_test, y_test) = processed_data[0]
    loss, accur = learn_and_test(x_train, y_train, x_val, y_val, x_test, y_test)
    print "Model loss after training: {}".format(loss)
    print "Model accuary after training: {}".format(accur)
else:
    all_loss = []
    all_accur = []
    for _ in range(len(processed_data)):
#     for _ in range(10):
        (x_train,y_train),(x_val, y_val),(x_test, y_test) = processed_data[_]
        loss, accur = learn_and_test(x_train, y_train, x_val, y_val, x_test, y_test)
        all_loss.append(loss)
        all_accur.append(accur)
    print "Model loss after training: {}".format(np.mean(all_loss))
    print "Model accuary after training: {}".format(np.mean(all_accur))

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])