import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, load_img, img_to_array
from tensorflow.keras.callbacks import CSVLogger, Callback
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import os
import random
import csv
import time

# DOE 2 : Factors: Image input size and batch size

TRAIN_DIR = '../dataset/train'
TEST_DIR = '../dataset/test'  
RESULTS_DIR = 'exp_batchsize_imgsize'

NUM_CLASSES = 7
EPOCHS = 80
PATIENCE = 10
TESTED_MODEL = 'EffNet'

FIXED_LR = 1e-4
FIXED_DEPTH = 20

BATCH_SIZES = {
    "Small": 32,
    "Medium": 64,
    "Large": 96
}

IMAGE_SIZES = {
    "Large": 112,
    "Small": 48
}

RUN_PER_CONFIG = 15

RESULTS_FILE = os.path.join(RESULTS_DIR, f"res_doe_2.csv")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)


class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []
    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()
    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def build_EfficientNetB0(img_size, learning_rate, unfreeze_depth):
    inputs = Input(shape=(img_size, img_size, 3))
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
    base_model.trainable = True 
    if unfreeze_depth == 0:
        base_model.trainable = False
    else:
        for layer in base_model.layers: layer.trainable = False
        for layer in base_model.layers[-unfreeze_depth:]: layer.trainable = True
            
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x) 
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def check_if_run_exists(img_size_val, batch_size_val):
    if not os.path.exists(RESULTS_FILE): return False
    try:
        df = pd.read_csv(RESULTS_FILE)
        matches = df[(df['Factor_Img_Size_Value'] == img_size_val) & 
                     (df['Factor_Batch_Size_Value'] == batch_size_val)]
        return not matches.empty
    except: return False

def initialize_csv():
    if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Run_Number",
                             "Factor_Img_Size_Label", 
                             "Factor_Img_Size_Value",
                             "Factor_Batch_Size_Label", 
                             "Factor_Batch_Size_Value",
                             "Blocking_Seed", 
                             "Response_Val_Accuracy",
                             "Time_Total_Sec", 
                             "Time_Avg_Epoch_Sec",
                             "Precision_Angry", 
                             "Precision_Disgust", 
                             "Precision_Fear", 
                             "Precision_Happy", 
                             "Precision_Sad", 
                             "Precision_Surprise", 
                             "Precision_Neutral"])

def append_result(res):
    with open(RESULTS_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([res["Run_Number"], 
                         res["Factor_Img_Size_Label"], 
                         res["Factor_Img_Size_Value"],
                         res["Factor_Batch_Size_Label"], 
                         res["Factor_Batch_Size_Value"],
                         res["Blocking_Seed"], 
                         res["Response_Val_Accuracy"],
                         res["Time_Total_Sec"], 
                         res["Time_Avg_Epoch_Sec"],
                         res.get("Precision_angry",0), 
                         res.get("Precision_disgust",0), 
                         res.get("Precision_fear",0),
                         res.get("Precision_happy",0), 
                         res.get("Precision_sad",0), 
                         res.get("Precision_surprise",0),
                         res.get("Precision_neutral",0)])


def run_doe():
    initialize_csv()
    
    total_configs = len(IMAGE_SIZES) * len(BATCH_SIZES) * RUN_PER_CONFIG

    current_run = 0
    
    for img_label, img_val in IMAGE_SIZES.items():
        for batch_label, batch_val in BATCH_SIZES.items():
            
            # Data are loaded one time per config
            train_datagen = ImageDataGenerator(
                preprocessing_function=preprocess_input
            )

            train_generator = train_datagen.flow_from_directory(
                TRAIN_DIR,
                target_size=(img_val, img_val),
                batch_size=batch_val,
                class_mode='categorical',
                shuffle=True
            )
            
            val_datagen = ImageDataGenerator(
                preprocessing_function=preprocess_input
            )
            val_generator = val_datagen.flow_from_directory(
                TEST_DIR,
                target_size=(img_val, img_val),
                batch_size=batch_val,
                class_mode='categorical',
                shuffle=False
            )
            
            for _ in range(RUN_PER_CONFIG):
                
                current_run += 1
                print(f"\n--- Run {current_run}/{total_configs} [Img:{img_val}, Batch:{batch_val}, Seed:{seed}] ---")
                
                if check_if_run_exists(img_val, batch_val):
                    print("--> Already done")
                    continue
                
                print("--> TODO")
                seed =random.randint(0, 1000)

                os.environ['PYTHONHASHSEED'] = str(seed)
                random.seed(seed)
                np.random.seed(seed)
                tf.random.set_seed(seed)


                model = build_EfficientNetB0(img_val, FIXED_LR, FIXED_DEPTH)
                
                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy', 
                    patience=PATIENCE, 
                    restore_best_weights=True, 
                    verbose=1
                    )
                
                time_cb = TimeHistory()
                
                log_file = os.path.join(RESULTS_DIR, f"log_run{current_run}_img{img_val}_batch{batch_val}.csv")
                csv_logger = CSVLogger(log_file, append=True)
                
                start = time.time()

                history = model.fit(
                    train_generator,
                    validation_data=val_generator,
                    epochs=EPOCHS,
                    batch_size=batch_val,
                    callbacks=[early_stop, csv_logger, time_cb],
                    verbose=1
                )
                
                duration = time.time() - start
                avg_epoch = np.mean(time_cb.times) if time_cb.times else 0
                best_acc = max(history.history['val_accuracy'])
                
                predictions = model.predict(val_generator)
                y_pred = np.argmax(predictions, axis=1)
                y_true = val_generator.classes
                
                class_labels = list(val_generator.class_indices.keys())
                report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
                
                res = {
                    "Run_Number": current_run,
                    "Factor_Img_Size_Label": img_label, 
                    "Factor_Img_Size_Value": img_val,
                    "Factor_Batch_Size_Label": batch_label, 
                    "Factor_Batch_Size_Value": batch_val,
                    "Blocking_Seed": seed, 
                    "Response_Val_Accuracy": best_acc,
                    "Time_Total_Sec": duration, 
                    "Time_Avg_Epoch_Sec": avg_epoch
                }
                for cls in class_labels: res[f"Precision_{cls}"] = report[cls]['precision']
                
                append_result(res)
                print(f"--> Done. Acc={best_acc:.4f}, Time={duration:.1f}s")
                
                del model
                tf.keras.backend.clear_session()

if __name__ == "__main__":
    run_doe()