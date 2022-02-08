import numpy as np
import scipy as sp
import pandas as pd
import librosa
from IPython.display import Audio
import os

''' This file contains code to process and test the ASV (attribute similarity vector) in a basic recommender engine.
'''

class ASV_Recommender:
    ''' A simple recommender to test the ASV (attribute similarity vector).
    Author: Sean Goldie
    Thesis Project, MM NYU 2022
    '''
    
    def __init__(self, audio_folder_path, csv_data_path, method):
        '''  
        Parameters:
        -----------
        audio_folder_path: string
            A string containing the folder to scan for audio files. 
            These filenames should be of the same format as the IDs
            in the first column of the csv_data_path.

        csv_data_path: string
            A string containing the path to the .csv file to use for
            song IDs and ASV data, as columns in that order.
            
        method: string
            The method to use for determining distance between vectors.
            Options:
                    "bhattacharyya" - Bhattacharyya coefficient
                    "euclidean" - euclidean distance
                    "cosine" - cosine similarity

        Returns:
        --------
        A new ASV_Recommender object.
        
        '''
        
        self.audio_folder_path = audio_folder_path
        self.csv_data_path = csv_data_path
        self.method = method
    
    
    def init(self):
        ''' Initialize the recommender
        
        Parameters:
        -----------
        None

        Returns:
        --------
        None
        
        '''
        
        data = pd.read_csv(self.csv_data_path)
        self.id_data = data.iloc[:,0].tolist()
        self.temp = data.iloc[:,1].tolist()
        self.recommendations = list()
    
    
    def prepare_asv_data(self, normalize=True):
        ''' Process the string ASV data into lists of floats.
        
        Parameters:
        -----------
        normalize: bool
            Whether or not to normalize the data to have mean of 0 and std of 1

        Returns:
        --------
        None
        
        '''
        self.asv_data = list()
        
        for i in range(len(self.temp)):
        
            temp = self.temp[i].strip("[]").split(',')
            asv = list()
            
            for i in temp:
                asv.append(float(i))
            
            self.asv_data.append(asv)
        
        self.asv_data = np.array(self.asv_data)
        
        if normalize:
            self.asv_data = (self.asv_data - np.mean(self.asv_data)) / np.std(self.asv_data)


    
    def reset_recommendations(self):
        ''' Reset the recommendations list (songs already recommended)
        
        Parameters:
        -----------
        None

        Returns:
        --------
        None
        
        '''
        
        self.recommendations = list()

        
    def already_recommended(self,index):
        ''' Check if a song has been recommended since the last time reset_recommendations() was called
        
        Parameters:
        -----------
        index: int
            index of the song to check

        Returns:
        --------
        Bool: whether or not the song has been recommended
        
        '''
        
        return index in self.recommendations

    
    def calculate_distance(self, asv1, asv2, method):
        ''' Calculate distance for two vectors.
        
        Parameters:
        -----------
        asv1: list
            the first asv to compare

        asv2: list
            the second one to compare
            
        method: string
            the method to use
            Options:
                    "bhattacharyya" - Bhattacharyya coefficient
                    "euclidean" - euclidean distance
                    "cosine" - cosine similarity

        Returns:
        --------
        coefficient: float
            Distance measurement

        Notes:
        ------
        Bhattacharyya coefficient: B(x, y) = sum_i sqrt(x[i] * y[i]) + sqrt((1 - x[i]) * (1 - y[i]))
        Euclidean distance: D(x, y) = sqrt(sum_i (y[i] - x[i])**2)
        Cosine similarity: S(x, y) = (sum_i x[i] * y[i]) / (sqrt(sum_i x[i]**2)) * (sqrt(sum_i y[i]**2))

        '''
        
        if method == "bhattacharyya":
            distance = 0
            for i in range(len(asv1)):
                distance += np.sqrt(asv1[i] * asv2[i]) + np.sqrt((1 - asv1[i]) * (1 - asv2[i]))
            return distance
            
        elif method == "euclidean":
            return np.linalg.norm(np.array(asv1) - np.array(asv2))
            
        elif method == "cosine":
            return sp.spatial.distance.cosine(asv2, asv1)

    
    def recommend_new_song(self, song_index):
        ''' Recommend a new song using the ASV.
        
        Parameters:
        -----------
        song_index: int
            the index of the song to compare against

        Returns:
        --------
        next_index: int
            the index of the recommendation

        '''
        
        if not self.already_recommended(song_index):
            self.recommendations.append(song_index)
        
        asv = self.asv_data[song_index]
        scores = np.empty( (len(self.asv_data)) )

        for i in range( len(scores) ):
            if i == song_index: pass    # Skip the song in question for analysis
            scores[i] = self.calculate_distance(self.asv_data[i], asv, self.method)

        next_index = np.argmax(scores)

        if self.already_recommended(next_index):
            while self.already_recommended(next_index):
                scores = scores[scores<np.max(scores)]
                next_index = np.argmax(scores)

        self.recommendations.append(next_index)
        
        return next_index

    
    def retrieve_song(self, index):
        ''' Retrieve a song in the dataset by its index.
        
        Parameters:
        -----------
        song_index: int
            the index of the song to retrieve

        Returns:
        --------
        file: np.array
            the song data for the next song
        
        fs: int
            the sample rate for the next song

        '''    

        file = ''.join(c for c in self.id_data[index] if c.isdigit())
        
        return librosa.load(self.audio_folder_path + '/' + file + '.mp3', sr=None, mono=False)
        
        