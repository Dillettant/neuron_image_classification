from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import random
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import *
from keras.optimizers import SGD, RMSprop
from keras import backend as K

def legacy_model_define(img_shape):
    ##############################################
    def euclidean_distance(inputs):
        assert len(inputs) == 2, \
            'Euclidean distance needs 2 inputs, %d given' % len(inputs)
        u, v = inputs
        return K.sqrt((K.square(u - v)).sum(axis=1, keepdims=True))

    def contrastive_loss(y, d):
        """ Contrastive loss from Hadsell-et-al.'06
            http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        """
        margin = 1
        return K.mean(y * K.square(d) + (1 - y) * K.square(K.maximum(margin - d, 0)))

    def create_pairs(x, digit_indices):
        """ Positive and negative pair creation.
            Alternates between positive and negative pairs.
        """
        pairs = []
        labels = []
        n = min([len(digit_indices[d]) for d in range(10)]) - 1
        for d in range(10):
            for i in range(n):
                z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
                pairs += [[x[z1], x[z2]]]
                inc = random.randrange(1, 10)
                dn = (d + inc) % 10
                z1, z2 = digit_indices[d][i], digit_indices[dn][i]
                pairs += [[x[z1], x[z2]]]
                labels += [1, 0]
        return np.array(pairs), np.array(labels)

    def create_base_network(in_dim):
        """ Base network to be shared (eq. to feature extraction).
        """
        seq = Sequential()
        seq.add(Dense(128, input_shape=(in_dim,), activation='relu'))
        seq.add(Dropout(0.1))
        seq.add(Dense(128, activation='relu'))
        seq.add(Dropout(0.1))
        seq.add(Dense(128, activation='relu'))
        return seq

    def compute_accuracy(predictions, labels):
        """ Compute classification accuracy with a fixed threshold on distances.
        """
        return labels[predictions.ravel() < 0.5].mean()
    ##############################################

    # network definition
    # create a Sequential for each element of the pairs
    input1 = Sequential()
    input2 = Sequential()
    input1.add(Layer(input_shape=img_shape))
    input2.add(Layer(input_shape=img_shape))

    # share base network with both inputs
    # G_w(input1), G_w(input2) in article
    base_network = create_base_network(in_dim)
    add_shared_layer(base_network, [input1, input2])

    # merge outputs of the base network and compute euclidean distance
    # D_w(input1, input2) in article
    lambda_merge = LambdaMerge([input1, input2], euclidean_distance)

    # create main network
    model = Sequential()
    model.add(lambda_merge)
    return model

def build_model(img_shape):
    # Shared Input Layer
    from keras.utils import plot_model
    from keras.models import Model
    from keras.layers import Input
    from keras.layers import Dense
    from keras.layers import Flatten
    from keras.layers.convolutional import Conv2D
    from keras.layers.pooling import MaxPooling2D
    from keras.layers.merge import concatenate
    # input layer
    visible = Input(shape=img_shape)
    # first feature extractor
    conv1 = Conv2D(32, kernel_size=4, activation='relu')(visible)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    flat1 = Flatten()(pool1)
    # second feature extractor
    conv2 = Conv2D(16, kernel_size=8, activation='relu')(visible)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    flat2 = Flatten()(pool2)
    # merge feature extractors
    merge = concatenate([flat1, flat2])
    # interpretation layer
    hidden1 = Dense(10, activation='relu')(merge)
    # prediction output
    output = Dense(1, activation='sigmoid')(hidden1)
    model = Model(inputs=visible, outputs=output)
    # summarize layers
    print(model.summary())
    return model

model = build_model()

batch_size = 16
# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms)
model.fit([pair_comp_0, pair_comp_1], y_labels, batch_size=batch_size, nb_epoch=nb_epoch,
          validation_data=([pair_comp_0, pair_comp_1], y_labels))

# compute final accuracy on training and test sets
pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(pred, tr_y)
pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(pred, te_y)