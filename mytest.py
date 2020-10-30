from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, ModelCheckpoint
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os


#initialize csv test name
row_list = [["filename", "meat", "veggie", "noodle"]]

for i in os.listdir("images"):
    row_list.append([i, 0, 0, 0])

with open('result.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(row_list)


BATCH_SIZE = 5
IMAGE_SIZE = (256,256)
dataframe = pd.read_csv('ranking_round/result.csv', delimiter=',', header=0)

datagen_noaug = ImageDataGenerator(rescale=1./255)

test_generator = datagen_noaug.flow_from_dataframe(
    dataframe=dataframe.loc[0:299],
    directory='ranking_round/images',
    x_col='filename',
    y_col=['meat', 'veggie', 'noodle'],
    shuffle=False,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='raw')

#Test Model
model = load_model('ranking_round/result.h5')
score = model.evaluate_generator(
    test_generator,
    steps=len(test_generator))
print('score (mse, mae):\n',score)

test_generator.reset()
predict = model.predict_generator(
    test_generator,
    steps=len(test_generator),
    workers = 1,
    use_multiprocessing=False)
# print('prediction:\n',predict)

for i in range(len(dataframe.loc[0:299])):
    dataframe.loc[i, 'meat'] = predict[i][0]*47
    dataframe.loc[i, 'veggie'] = predict[i][1]*101
    dataframe.loc[i, 'noodle'] = predict[i][2]*268

dataframe.to_csv (r"result.csv", index=False, header=True)