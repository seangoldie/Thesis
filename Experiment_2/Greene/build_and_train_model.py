import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D, Input
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model

tf.compat.v1.disable_eager_execution()

""" Current experiment design:
* Data shape is 128 x 1280 x 1
* Model is:
    Conv2D(filters=64, kernel_size=4, input_shape=(128,1280,1))
    MaxPooling2D(pool_size=(4, 4)
    Dropout(0.5)
    Conv2D(filters=128, kernel_size=4, input_shape=(64,640,1))
    MaxPooling2D(pool_size=(4, 4)
    Dropout(0.5)
    Conv2D(filters=256, kernel_size=4, input_shape=(32,320,1))
    MaxPooling2D(pool_size=(2, 2)
    Dropout(0.5)
    Flatten()
"""

def load_the_data(data_pathway, labels_pathway):
    data_folder = os.listdir(data_pathway)
    labels_folder = os.listdir(labels_pathway)

    label_len = len(labels_folder)
    data_len = len(data_folder)

    print("Number of samples present in the labels folder: ", label_len)
    print("Number of samples present in the data folder: ", data_len)
    
    # data = np.empty((length, 128, 1280, 1))
    data = np.empty((label_len, 128, 1280))
    labels = np.empty((label_len, 161))
    deletes = list() # Keep track of which relevant indices generated errors and remove them from the arrays

    for i in range(label_len):
        try:
            labels[i] = np.load(
                labels_pathway + labels_folder[i]
            )
            temp = np.load(
                data_pathway + data_folder[
                    data_folder.index(
                        labels_folder[i]
                    )
                ] # Order the data
            )
            if temp.shape != (128,1280):
                data[i] = temp[:128,:1280]
            else:
                data[i] = temp          
        except Exception as e: # Keep track of the indices that create problems
            print(e)
            deletes.append(i)
    
    # Delete the indices from the arrays that created problems
    np.delete(
        labels, 
        np.array(deletes, dtype=np.uint),
        axis=0 
    )
    np.delete(
        data,
        np.array(deletes, dtype=np.uint),
        axis=0
    )
    print("Removing empty rows from labels")
    labels = labels[labels != np.empty((1, 161))]
    print("Number of labels removed is ", len(labels) - label_len)
    print("Removing empty rows from data")
    data = data[data != np.empty((1, 128, 1280))]
    print("Number of data removed is ", len(data) - data_len)

    return data, labels, deletes

def main():

    DATA_PATH = "/scratch/smg8196/spectrogram_data_train/"
    LABELS_PATH = "/scratch/smg8196/spectrogram_labels_train/"
    BATCH_SIZE = 64

    X_train, y_train, deletes = load_the_data(DATA_PATH, LABELS_PATH)

    print("Number of deleted samples:", len(deletes))

    EPOCHS = 20
    OUTPUT_PATH = "/scratch/smg8196/Experiment_2/Greene/trained_model/"
    if not os.path.exists(OUTPUT_PATH): 
        os.mkdir(OUTPUT_PATH)

    # Build the model
    _input = Input(shape=(128,1280,1))
    x = Conv2D(filters=64, kernel_size=4, input_shape=(128,1280,1), data_format="channels_last")(_input)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(4, 4))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(filters=128, kernel_size=4, input_shape=(64,640,1), data_format="channels_last")(x) # changed from _input
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(4, 4))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(filters=256, kernel_size=4, input_shape=(32,320,1), data_format="channels_last")(x) # changed from _input
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    output = Dense(161, activation = 'sigmoid')(x)
    model = Model(_input, output)
    model.compile(optimizer=optimizers.Adam(), loss="binary_crossentropy")

    print(model.summary())

    # Construct save pathways
    model_spec = tf.keras.utils.serialize_keras_object(model)
    with open( os.path.join(OUTPUT_PATH, "model_spec.pkl"), "wb") as fd:
        pickle.dump(model_spec, fd)
    weight_path = os.path.join(OUTPUT_PATH, "epoch_{epoch:02d}_checkpoint.h5")
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = weight_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
        save_freq="epoch"
    )

    model.fit(
        x=X_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        callbacks=[model_checkpoint_callback],
        validation_split=0.2,
        shuffle=True,
        epochs=EPOCHS,
        workers=1,
        verbose=1
    )

if __name__ == "__main__":
    main()
