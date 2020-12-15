# -*- coding:utf-8 -*-

import datetime, os
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage import img_as_float
from sklearn.metrics import roc_auc_score
import pandas as pd
import tensorflow.keras.applications.resnet as resnet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from PIL import ImageFile
import argparse
import glob


def preprocess(args):
    # read csv
    print(glob.glob(f"{args.input_path}/*/{args.csv_name}"))
    csv_path = glob.glob(f"{args.input_path}/*/{args.csv_name}")[0]
    df = pd.read_csv(csv_path)

    # create label data
    columns = [col for col in df.columns if col != "image_id"]
    with open(f"{args.preprocessed_data_path}/labels.txt", "w") as f:
        f.write("\n".join(columns))

    # split train and test datas
    train_size = int(df.shape[0] * 0.80)
    df_train = df[:train_size]
    df_test = df[train_size:].reset_index()
    df_test = df_test.drop("index", axis=1)

    # preprocessing for train data
    for idx in tqdm(range(df_train.shape[0])):
        file_name = df_train["image_id"][idx]
        file_path = glob.glob(f"{args.input_path}/*/{file_name}.jpg")[0]
        img = imread(file_path)
        img = resize(img, (400, 400))
        imsave(f"{args.preprocessed_data_path}/{file_name}.png", img, check_contrast=False)
        break
    df_train.to_csv(f"{args.preprocessed_data_path}/train.csv", index=False)

    # preprocessing for test data
    for idx in tqdm(range(df_test.shape[0])):
        file_name = df_test["image_id"][idx]
        file_path = glob.glob(f"{args.input_path}/*/{file_name}.jpg")[0]
        img = imread(file_path)
        img = resize(img, (400, 400))
        imsave(f"{args.preprocessed_data_path}/{file_name}.png", img, check_contrast=False)
        break
    df_test.to_csv(f"{args.preprocessed_data_path}/test.csv", index=False)

def train(args):
    # prepare data
    df_train = pd.read_csv(f"{args.preprocessed_data_path}/train.csv")
    df_test = pd.read_csv(f"{args.preprocessed_data_path}/test.csv")
    with open(f"{args.preprocessed_data_path}/labels.txt") as f:
        labels = f.read().splitlines()
    X_train = np.array([
        imread(f"{args.preprocessed_data_path}/{filename}.png")
        for filename in df_train["image_id"]
    ])
    X_train = X_train / 255
    X_test = np.array([
        imread(f"{args.preprocessed_data_path}/{filename}.png")
        for filename in df_test["image_id"]
    ])
    X_test = X_test / 255
    y_train = df_train[labels].values
    y_test = df_test[labels].values

    # augumentation setting
    datagen = image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)

    # prepare model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X_train[0].shape))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.8))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(labels), activation='softmax'))

    model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy',
        metrics=['accuracy', AUC()]
        )

    # callback setting
    mc = ModelCheckpoint(
        filepath=f"{args.output_path}/model_weight.h5",
        save_best_only=True,
        save_weights_only=False
    )
    tb = TensorBoard(
        log_dir=args.log_path,
        histogram_freq=1
    )
    callbacks = [mc, tb]

    # fit
    datagen.fit(X_train)
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
                        steps_per_epoch=len(X_train) / 32, epochs=100,
                        validation_data=(X_test, y_test), 
                        callbacks=callbacks)

if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='aqualium demo file')
    parser.add_argument('running_parten', type=str, help='both | train | preprocess')
    parser.add_argument('--input_path', default='/kqi/input/training')
    parser.add_argument('--output_path', default='/kqi/output/demo')
    parser.add_argument('--preprocessed_data_path', default='/kqi/output/preprocessed_data')
    parser.add_argument('--log_path', default='/kqi/output/demo/log')
    parser.add_argument('--csv_name', default='train.csv')
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.preprocessed_data_path, exit_ok=True)
    os.makedirs(args.log_path, exist_ok=True)

    if args.running_parten == "both":
        preprocess(args)
        train(args)