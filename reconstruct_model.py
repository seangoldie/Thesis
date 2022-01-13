import pickle
import sys
import tensorflow as tf
import os


def main(EPOCH):
    
    '''Reconstructs and saves a model from its specs and weights. Don't forget to pass
    the int of the epoch you wish to reconstruct!
    '''

    stem = "/scratch/smg8196/thesis_project_local_new/experiment_2_trained_model/"
    output_path = "/scratch/smg8196/thesis_project_local_new/experiment_2_trained_model_reconstructed/"
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    spec_filepath = stem + "model_spec.pkl"
    weights_filepath = stem + f"epoch_{EPOCH}_checkpoint.h5"
    with open(spec_filepath, 'rb') as file:
        spec = pickle.load(file)
        model = tf.keras.models.model_from_config(spec)
        model.load_weights(weights_filepath)
    
    model.save(output_path + f"_epoch_{EPOCH}")

if __name__ == "__main__":
    main(sys.argv[1])
