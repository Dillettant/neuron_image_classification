import numpy as np
import keras

def ImageNet_preprocessing(data_x, input_img_size=(256, 256, 2), output_img_size=(256, 256, 3)):
    new_data_x = np.resize(data_x, (len(data_x), 256, 256, 3))
    return new_data_x

def read_data():
    # write a function to select x_train y_train x_test y_test
    data_x = np.load('image_data_256x256.npy')
    data_y = np.load('labels.npy')

    # concatenate together and shuffle the data
    data_x = np.swapaxes(np.swapaxes(data_x,1,2),2,3)
    data_y = np.array([data_y]).T

    # Expand 2 channels into 3 channels
    # data_x = ImageNet_preprocessing(data_x)

    # shuffle the data first
    from sklearn.utils import shuffle
    data_x, data_y = shuffle(data_x, data_y)

    return data_x, data_y

def cross_validataion_splits(data_x, data_y, test_set_size = 0.2, val_set_size = 0.2, cv_split_size = 10, num_classes = 3):
    from sklearn.cross_validation import train_test_split
    from sklearn.model_selection import KFold
    processed_data = []
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
    print("Cross validation data has been split!")
    return processed_data


def main():
    data_x, data_y = read_data()
    processed_data = cross_validataion_splits(data_x, data_y)
    import pickle
    pickle.dump(processed_data, open("./saved_data/processed_data.p", "wb"))  # save it into a file named save.p

if __name__ == '__main__':
    main()