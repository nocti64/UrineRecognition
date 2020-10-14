from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense
import keras
import os
from keras.utils import np_utils
import tensorflow as tf
from keras.models import Model, load_model
import numpy as np
# define data generator
class DataGenerator(keras.utils.Sequence):
    def __init__(self, batch_size, file_folder, shuffle=True):
        self.batch_size=batch_size
        self.file_folder = file_folder
        self.list_IDs = self.get_all_imgs_path()
        self.shuffle = shuffle
        self.labels_dic = self.get_labels()  # int key, string value
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))
   
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # get batch imgs path
        batch_imgs_path = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(batch_imgs_path)

        return X, y
    
    def get_labels(self):
        # get labels
        labels = {}
        with open(self.file_folder+"/labels.csv") as f:
            for row in f:
                row = row.replace("\n",'').split(',')
                labels[int(row[0])]=row[1]
            return labels
    
    def get_all_imgs_path(self):
        # 回傳相對路徑，只有圖片檔名
        data_path = self.file_folder+'/images'
        return os.listdir(data_path)
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
  
    def one_hot_encode(self, l):
        label= np.array([0,0,0,0,0])
        label[int(l)]=1
        return label
        
    def __data_generation(self, batch_imgs_path):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        
        X =[]
        y =[]
        
        for img in batch_imgs_path:
            image_path = self.file_folder+"/images/"+img # 這裡才把絕對路徑接上去
            image = tf.keras.preprocessing.image.load_img(image_path)
            input_arr = keras.preprocessing.image.img_to_array(image)
            X.append(input_arr)
            idx = img[:-4]
            y.append(self.one_hot_encode(self.labels_dic[int(idx)]))
   
        X = np.array(X)  # (bs, h, w, c)
        y = np.array(y)

        return X, y

def create_model():
    IRNv2 = InceptionResNetV2(include_top=False, pooling='avg')
    dense_layer0 = Dense(100, activation='relu')(IRNv2.output)
    dense_layer1 = Dense(25, activation='relu')(dense_layer0)
    outputs = Dense(5, activation='softmax')(dense_layer1)
    UrNet = Model(IRNv2.inputs, outputs)
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    UrNet.compile(optimizer=opt, loss='categorical_crossentropy',metrics='accuracy')
    return UrNet

if __name__=='__main__()':
    batch_size = 8
    epochs = 10
    train_dir = './Urine/training_data'  
    val_dir = './Urine/val_data'

    # create data generator
    trainGen = DataGenerator(batch_size=batch_size, image_data_generator=imgGen, file_folder=train_dir)
    valGen = DataGenerator(batch_size=batch_size, image_data_generator=imgGen, file_folder=val_dir)

    # get model
    UrNet = create_model()
    his = UrNet.fit(trainGen, steps_per_epoch=50//2, epochs=epochs, validation_data=valGen, verbose=1)

