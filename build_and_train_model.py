import numpy as np
import os
import pickle
import tensorflow as tf
import shutil
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D, Input
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model

tf.compat.v1.disable_eager_execution()

def load_the_data(data_pathway, labels_pathway):
    data_folder = os.listdir(data_pathway)
    labels_folder = os.listdir(labels_pathway)

    length = len(labels_folder)
    
    data = np.empty((length, 128, 640, 1)) # Cropped from 128 by 1280
    labels = np.empty((length, 161))
    deletes = list() # Keep track of which relevant indices generated errors and remove them from the arrays

    for i in range(length):
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
            if temp.shape == (128,1280): # Only the data with the right dimensions gets cropped and added
                data[i] = temp[:,:640] # Crop the data
        except: # Keep track of the indices that create problems
            deletes.append(i)
    
    # Delete the indices from the arrays that created problems
    np.delete( labels, np.array(deletes, dtype=np.uint), axis=0 )
    np.delete( data, np.array(deletes, dtype=np.uint), axis=0 )

    return data, labels, deletes

# def custom_data_generator(
#     validate_split
#     data,
#     labels,
#     batch_size,
#     set
#     ):

#     length = len(labels)
#     num_batches = int(np.ceil(length/batch_size))

#     sp = np.round(length * validate_split)
#     X_train = data[0:sp-1]
#     X_valid = data[sp:length-1]
#     y_train = labels[0:sp-1]
#     y_valid = labels[sp:length-1]

#     X_train = tf.split()

#     if set == "train":
#         for i in range(num_batches):
#             j = i * batch_size
#             k = j + batch_size
#             if k >= length:
#                 k = length
#             yield X_train[j:k], y_train[j:k]
    
#     elif set == "validate":
#         for i in range(num_batches):
#             j = i * batch_size
#             k = j + batch_size
#             if k >= length:
#                 k = length
#             yield X_valid[j:k], y_valid[j:k]


def main():

    DATA_PATH = "/scratch/smg8196/spectrogram_data_train/"
    LABELS_PATH = "/scratch/smg8196/spectrogram_labels_train/"
    BATCH_SIZE = 64

    X_train, y_train, deletes = load_the_data(DATA_PATH, LABELS_PATH)

    print("Number of deleted samples:", len(deletes))

    EPOCHS = 20
    OUTPUT_PATH = "/scratch/smg8196/thesis_project_local_new/experiment_2_trained_model/"
    if not os.path.exists(OUTPUT_PATH): 
        os.mkdir(OUTPUT_PATH)

    # Build the model
    _input = Input(shape=(128,640,1))
    x = Conv2D(filters=64, kernel_size=4, input_shape=(128,640,1), data_format="channels_last")(_input)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(4, 4))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(filters=128, kernel_size=4, input_shape=(64,320,1), data_format="channels_last")(x) # changed from _input
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(4, 4))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(filters=256, kernel_size=4, input_shape=(32,160,1), data_format="channels_last")(x) # changed from _input
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

    # Train the model
#     train_generator = custom_data_generator(
#         0.2,
#         data,
#         labels,
#         64,
#         "train"
#     )

#     valid_generator = custom_data_generator(
#         0.2,
#         data,
#         labels,
#         64,
#         "validate"
# )
    
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
        workers=16,
        verbose=1
        )

if __name__ == "__main__":
    main()
