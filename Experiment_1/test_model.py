import numpy as np
import os
import tensorflow as tf
import pandas as pd

def load_the_data(data_pathway, labels_pathway):

    data_folder = os.listdir(data_pathway)
    labels_folder = os.listdir(labels_pathway)

    length = len(labels_folder)
    
    data = np.empty((length, 128, 640, 1)) # Cropped from 128 by 1280
    labels = np.empty((length, 161))
    deletes = list()
    filenames = list()

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

            filenames.append( # If all else worked, append the filename in order
                labels_folder[i]
            )

        except: # Keep track of the indices that create problems
            deletes.append(i)
    
    # Delete the indices from the arrays that created problems
    np.delete( labels, np.array(deletes, dtype=np.uint), axis=0 )
    np.delete( data, np.array(deletes, dtype=np.uint), axis=0 )

    return data, labels, filenames, deletes

def main():

    DATA_PATH = "/scratch/smg8196/spectrogram_data_test/"
    LABELS_PATH = "/scratch/smg8196/spectrogram_labels_test/"
    LABELS_SAVE_PATH = "/scratch/smg8196/experiment_2_labels/"
    DATA_SAVE_PATH = "/scratch/smg8196/experiment_2_results/"
    TRAINED_MODEL_PATH = "/scratch/smg8196/thesis_project_local_new/experiment_2_trained_model_reconstructed/_epoch_20"

    if not os.path.exists(LABELS_SAVE_PATH):
        os.mkdir(LABELS_SAVE_PATH)
    if not os.path.exists(DATA_SAVE_PATH):
        os.mkdir(DATA_SAVE_PATH)

    X_test, y_test, filenames, deletes = load_the_data(DATA_PATH, LABELS_PATH)

    model = tf.keras.models.load_model(TRAINED_MODEL_PATH)

    print("Number of deleted samples:", len(deletes))

    np.save(LABELS_SAVE_PATH + 'test_labels.npy', y_test)

    p = model.predict(X_test)

    predictions = list()

    for i in p:
        asv = list()
        for j in i:
            asv.append(j)
        predictions.append(asv)
    
    results = pd.DataFrame({"Filename": filenames, "Predictions": predictions})
    results.to_csv(DATA_SAVE_PATH + "test_results.csv", index=False)

if __name__ == "__main__":
    main()
