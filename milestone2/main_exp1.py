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
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import random
import csv

# DOE 1 : Factors: Learning rate and Unfrozen Layer Depth

TRAIN_DIR = '../dataset/train'
TEST_DIR = '../dataset/test'

IMG_SIZE = 112
NUM_CLASSES = 7
BATCH_SIZE = 32
EPOCHS = 80 
PATIENCE = 10

LEARNING_RATES = {
    "Low": 1e-5, 
    "High": 1e-4
}

UNFREEZE_DEPTHS = {
    "50": 50, 
    "20": 20
}

RUN_PER_CONFIG = 15

RESULTS_FILE = f"res_doe_1.csv"


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

val_generator = val_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

def build_EfficientNetB0(learning_rate, unfreeze_depth):
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
    base_model.trainable = True 
    
    if unfreeze_depth == 0:
        base_model.trainable = False
    else:
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[-unfreeze_depth:]:
            layer.trainable = True
            
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x) 
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def check_if_run_exists(lr_val, depth_val):
    if not os.path.exists(RESULTS_FILE):
        return False
    try:
        df = pd.read_csv(RESULTS_FILE)
        matches = df[
            (df['Factor_A_LR_Value'] == lr_val) & 
            (df['Factor_B_Depth_Value'] == depth_val)
        ]
        return not matches.empty
    except pd.errors.EmptyDataError:
        return False

def initialize_csv():
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Factor_A_LR_Label", "Factor_A_LR_Value",
                "Factor_B_Depth_Label", "Factor_B_Depth_Value",
                "Blocking_Seed", "Response_Val_Accuracy"
            ])

def append_result_to_csv(result_dict):
    with open(RESULTS_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            result_dict["Factor_A_LR_Label"],
            result_dict["Factor_A_LR_Value"],
            result_dict["Factor_B_Depth_Label"],
            result_dict["Factor_B_Depth_Value"],
            result_dict["Blocking_Seed"],
            result_dict["Response_Val_Accuracy"]
        ])

# ==========================================
# 5. MAIN LOOP
# ==========================================
def run_doe():
    initialize_csv()

    total_runs = len(LEARNING_RATES) * len(UNFREEZE_DEPTHS) * RUN_PER_CONFIG
    current_run_check = 0
    
    for lr_name, lr_val in LEARNING_RATES.items():
        for depth_name, depth_val in UNFREEZE_DEPTHS.items():
            
            current_run_check += 1
            print(f"\n--- Check Run {current_run_check}/{total_runs} ---")
            print(f"Config: LR={lr_name}, Depth={depth_name}")
            
            if check_if_run_exists(lr_val, depth_val):
                print("--> Already done")
                continue
            
            print("--> TODO")
            
            seed =random.randint(0, 1000)
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)
            
            model = build_EfficientNetB0(lr_val, depth_val)
            
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', 
                patience=PATIENCE, 
                restore_best_weights=True,
                verbose=1
            )
            log_suffix = f"depth_{depth_val}_lr_{lr_val}"
            csv_logger = CSVLogger(
                f"training_log_{log_suffix}_run_{current_run_check}.csv",
                separator=",",
                append=True
            )
            
            history = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=EPOCHS,
                callbacks=[early_stop, csv_logger],
                verbose=1
            )
            
            best_val_acc = max(history.history['val_accuracy'])
            print(f"--> Res : Acc = {best_val_acc:.4f}")
            
            result_data = {
                "Factor_A_LR_Label": lr_name,
                "Factor_A_LR_Value": lr_val,
                "Factor_B_Depth_Label": depth_name,
                "Factor_B_Depth_Value": depth_val,
                "Blocking_Seed": seed,
                "Response_Val_Accuracy": best_val_acc
            }
            append_result_to_csv(result_data)
            
            del model
            tf.keras.backend.clear_session()
                
if __name__ == "__main__":
    run_doe()