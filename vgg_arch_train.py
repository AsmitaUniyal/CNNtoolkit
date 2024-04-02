import numpy as np
from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras import backend as K
from keras.utils import np_utils
from scipy import misc
from keras import optimizers
import cv2
import os

TH_WEIGHTS_PATH = 'rcmalli_vggface_th_weights_th_ordering.h5'
TH_WEIGHTS_PATH_NO_TOP = '/home/uia70982/face_reco/rcmalli_vggface_th_weights_th_ordering_notop.h5'

num_of_persons = 7


def train():

    # Load image
    train = []
    test = []
    full_data = []
    sample_number = 0
    train_label = []                                               #np.zeros((900 * num_of_persons,), dtype=int)
    test_label = []                                                 #np.zeros((600 * num_of_persons,), dtype=int)
    train_count = 0
    test_count = 0
    j=0

    for j in range(num_of_persons):  # no. of persons
        print ("Person no: ", j)
        data_loc= "/home/uia70982/face_reco/auto_image_save/person_%d" % (j + 1)
        print(data_loc)
        data_list = os.listdir(data_loc)
        count = 0
        for i in data_list:  # no. of samples
            # Train:0-5     Test:6-9
            count= count+1
            #input_path = "/mnt/disk1/vijay/database/real_time_images/person_%d/%d.png" % (j + 1, i)
            #input_path = "D:/dataset/auto_image_save/person %d/%d.png" % (j + 1, i)
            input_image = cv2.imread(data_loc + '/' + i)
            im = np.zeros((224,224,3), np.float32)
            #input_image = cv2.imread(input_path)

			# Converting to grayscale and making it three channel
            gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
            resize_image = cv2.resize(gray_image,(224,224))

            im[:,:,0] = resize_image
            im[:,:,1] = resize_image
            im[:,:,2] = resize_image

            im[:, :, 0] -= 93.5940
            im[:, :, 1] -= 104.7624
            im[:, :, 2] -= 129.1863

            im = np.transpose(im, (2, 0, 1))
            im = np.expand_dims(im, axis=0)
            im = im.flatten()

            if ((count % 10) <= 6):
                train.append(im)
                train_label.append(j)
                #train_count = train_count + 1
            else:
                test.append(im)
                test_label.append(j)
                #test_count = test_count + 1
            full_data.append(im)

        sample_number = sample_number + 1

    train = np.array(train)
    train = train.reshape(train.shape[0], 3, 224, 224)
    print (train.shape)

    test = np.array(test)
    test = test.reshape(test.shape[0], 3, 224, 224)
    print (test.shape)

    train = train.astype('float32')
    test = test.astype('float32')
    train /= 255
    test /= 255

    train_label = np_utils.to_categorical(train_label, num_of_persons)
    test_label = np_utils.to_categorical(test_label, num_of_persons)
    print("label",train_label.shape)
    print(test_label.shape)
    # Load model parameters
    hidden_dim = 512

    image_input = Input(shape=(3, 224, 224))

    vgg_model = VGGFace(image_input,False, None) # paramaters:input_tensor,include_top,weights
    vgg_model.load_weights(TH_WEIGHTS_PATH_NO_TOP)

    last_layer = vgg_model.get_layer('pool5').output

    x = Flatten(name='flatten')(last_layer)
    x = Dense(hidden_dim, activation='relu', name='fc6')(x)
    x = Dropout(0.5)(x)
    x = Dense(hidden_dim, activation='relu', name='fc7')(x)
    x = Dropout(0.5)(x)
    out = Dense(num_of_persons, activation='softmax', name='fc8')(x)

    custom_vgg_model = Model(image_input, out)
    custom_vgg_model.summary()

    layer_count = 0
    for layer in custom_vgg_model.layers:
        layer_count = layer_count + 1

    for l in range(layer_count - 5):
        custom_vgg_model.layers[l].trainable = False

    #sgd = optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0004, nesterov=False)
    custom_vgg_model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])
    print("Compiled")
    custom_vgg_model.fit(train, train_label, batch_size=32, nb_epoch=1, verbose=1)
    (loss, accuracy) = custom_vgg_model.evaluate(test, test_label, batch_size=10, verbose=1)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

    custom_vgg_model.save('transfer_learning-adadelta-7-ppl_gray.h5')

def VGGFace(input_tensor,include_top, weights):
    # paramaters:input_tensor,include_top,weights

    if not K.is_keras_tensor(input_tensor):
        img_input = Input(tensor=input_tensor)
    else:
        img_input = input_tensor

    # Block 1
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='conv1_1', dim_ordering='th')(img_input)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='conv1_2', dim_ordering='th')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='conv2_1', dim_ordering='th')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='conv2_2', dim_ordering='th')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2', dim_ordering="th")(x)

    # Block 3
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_1', dim_ordering='th')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_2', dim_ordering='th')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_3', dim_ordering='th')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3', dim_ordering="th")(x)

    # Block 4
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_1', dim_ordering='th')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_2', dim_ordering='th')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_3', dim_ordering='th')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4', dim_ordering="th")(x)

    # Block 5
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv5_1', dim_ordering='th')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv5_2', dim_ordering='th')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv5_3', dim_ordering='th')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5', dim_ordering="th")(x)

    # Create model
    model = Model(img_input, x)
    model.summary()

    return model

if __name__ == '__main__':
    train()