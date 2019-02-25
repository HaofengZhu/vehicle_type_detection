import os
import pandas as pd
import numpy as np
from skimage import io,transform
import os, sys
from tqdm import tqdm
from keras import applications
from keras.models import Model
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
## set path for images
import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TRAIN_PATH = './trainData'
TEST_PATH = './trainData'
img_size=(256,256,3)

def getDirLabels(path):
    dirList = []
    # 返回一个列表，其中包含在目录条目的名称(google翻译)
    files = os.listdir(path)
    # 先添加目录级别
    for f in files:
        if(os.path.isdir(path + '/' + f)):
            # 排除隐藏文件夹。因为隐藏文件夹过多
            if(f[0] == '.'):
                pass
            else:
                # 添加非隐藏文件夹
                dirList.append(f)
    # 当一个标志使用，文件夹列表第一个级别不打印
    return dirList

def getFiles(path):
    fileList = []
    files = os.listdir(path)
    for f in files:
        if(os.path.isfile(path + '/' + f)):
            # 添加文件
            fileList.append(f)
    return  fileList

def read_img(img_path):
    img = io.imread(img_path)
    img=transform.resize(img,img_size)
    return img
def getData(path):
    #path为数据的根目录，下级目录为各类图片的子目录
    all_dir=getDirLabels(path)
    data,labels=[],[]
    for dir in all_dir:
       fileList=getFiles(os.path.join(path,dir))
       for f in fileList:
           data.append(read_img(os.path.join(path,dir,f)))
           labels.append(dir)
    return data,labels

## import libaries

## load data
train_data,train_labels = getData(TRAIN_PATH)
test_data,test_labels = getData(TEST_PATH)

# normalize images
x_train = np.array(train_data, np.float32) / 255.
x_test = np.array(test_data, np.float32) / 255.

#将labels转换成enum
Y_train = {k:v+1 for v,k in enumerate(set(train_labels))}
y_train = [Y_train[k] for k in train_labels]
y_train = np.array(y_train)

#存储label及其对应的编号no
labels=[]
for k,v in Y_train.items():
    labels.append([k,v])
pd.DataFrame(labels,columns=['label','no']).to_csv('labels_match.csv',index=0)

Y_test = {k:v+1 for v,k in enumerate(set(test_labels))}
y_test = [Y_test[k] for k in test_labels]
y_test = np.array(y_test)


y_train = to_categorical(y_train)

#Transfer learning with Inception V3
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=img_size)

## set model architechture
add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dense(y_train.shape[1], activation='softmax'))

model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
            metrics=['accuracy'])

model.summary()

batch_size = 1 # tune it
epochs = 30 # increase it
train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)
train_datagen.fit(x_train)

history = model.fit_generator(
    train_datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=x_train.shape[0] // batch_size,
    epochs=epochs,
    callbacks=[ModelCheckpoint('VGG16-transferlearning2.model', monitor='val_acc', save_best_only=True)]
)

## predict test data
predictions = model.predict(x_test)

# get labels
predictions = np.argmax(predictions, axis=1)
rev_y = {v:k for k,v in Y_train.items()}
pred_labels = [rev_y[k] for k in predictions]

print(test_labels)
print(pred_labels)
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=batch_size)
print(classification_report(test_labels,pred_labels))

model.save('keras_vgg16.h5')
