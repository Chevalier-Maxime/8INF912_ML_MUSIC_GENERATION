# Classifier that allow us to validate our results
# 3 differents classes : Final Fantasy, Real Music, Random Music

# Problem : the output oof our generator is only on one track and
#           the input is on multiple track with multiple 
#           instruments.
# Solution : Preprocess all files with our own preprocessor

# Inspired by https://github.com/sandershihacker/midi-classification-tutorial/blob/master/midi_classifier.ipynb

from argparse import ArgumentParser
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import pretty_midi
import warnings
import os
from preprocess import EmptyDataFolderException
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('classifier')

def get_genres(path):
    """
    Get the differents genres from folder names
    @input path : The path of the input folder containing genre's subfolder
    @type path: String

    @return: A pandas dataframe containing the genres and midi path associated
    @rtype: pandas.DataFrame 
    """

    root_dir = os.listdir(path)
    if len(root_dir) == 0:
        raise EmptyDataFolderException()
    
    midi_path = []
    genres = []

    logger.info('Entering get_genre function')
    for node in root_dir:
        node_absolute = os.path.join(path, node)
        if os.path.isdir(node_absolute):
            game_dir = os.listdir(node_absolute)
            logger.info('Processing genre : %s' % node)

            if len(game_dir) == 0:
                logger.warning('No files in %s' % node_absolute)
            for node2 in game_dir:
                # Too much log info logger.info('Processing midi file : %s' %node2)
                # Check extension
                ext = os.path.splitext(node2)[1]
                node2_absolute = os.path.join(node_absolute, node2)

                if ext.lower() == '.mid':
                    genres.append(node)
                    midi_path.append(node2_absolute)
                else:
                    logger.warning('%s is not a *.mid file' % node2_absolute)
        else:
            logger.warning('%s should be a folder' % node_absolute)
    df = pd.DataFrame({"Path": midi_path, "Genre": genres})
    return df
    

def normalize_features(features):
    """
    This function normalizes the features to the range [-1, 1]
    
    @input features: The array of features.
    @type features: List of float
    
    @return: Normalized features.
    @rtype: List of float
    """
    tempo = (features[0] - 150) / 300
    num_sig_changes = (features[1] - 2) / 10
    resolution = (features[2] - 260) / 400
    time_sig_1 = (features[3] - 3) / 8
    time_sig_2 = (features[4] - 3) / 8
    return [tempo, resolution, time_sig_1, time_sig_2]


def get_features(path):
    """
    This function extracts the features from a midi file when given its path.
    
    @input path: The path to the midi file.
    @type path: String
    
    @return: The extracted features.
    @rtype: List of float
    """
    try:
        # Test for Corrupted Midi Files
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            file = pretty_midi.PrettyMIDI(path)
            
            tempo = file.estimate_tempo()
            num_sig_changes = len(file.time_signature_changes)
            resolution = file.resolution
            ts_changes = file.time_signature_changes
            ts_1 = 4
            ts_2 = 4
            if len(ts_changes) > 0:
                ts_1 = ts_changes[0].numerator
                ts_2 = ts_changes[0].denominator
            return normalize_features([tempo, num_sig_changes, resolution, ts_1, ts_2])
    except:
        logger.warning('corrupted midi file')
        return None


def extract_midi_features(path_df):
    """
    This function takes in the path DataFrame, then for each midi file, it extracts certain
    features, maps the genre to a number and concatenates these to a large design matrix to return.
    
    @input path_df: A dataframe with paths to midi files, as well as their corresponding matched genre.
    @type path_df: pandas.DataFrame
    
    @return: A matrix of features along with label.
    @rtype: numpy.ndarray of float
    """
    all_features = []
    logger.info('Start extraction of features')
    logging_info_nbstep = len(path_df.index)-1
    for index, row in path_df.iterrows():
        logger.info('Step %d/%d' % (index,logging_info_nbstep))
        features = get_features(row.Path)
        genre = label_dict[row.Genre]
        if features is not None:
            features.append(genre)
            all_features.append(features)
    return np.array(all_features)



if __name__ == "__main__":
    parser = ArgumentParser()

    # Mandatory arguments
    parser.add_argument('input_folder', type=str, help='The folder containing \
            the game folders containing *.mid (MIDI) files')
    parser.add_argument('output_folder', type=str, help='The folder that will \
    #        be provided to the network')

    # Optional arguments
    parser.add_argument('--provideFeatures', type=str, help='Provide path to stored feature .pkl file')

    args = parser.parse_args()

    dataframe = get_genres(args.input_folder)
    print(dataframe.head())

    # Create Genre List and Dictionary
    label_list = list(set(dataframe.Genre))
    label_dict = {lbl: label_list.index(lbl) for lbl in label_list}

    if args.provideFeatures:
        labeled_features = np.load(args.provideFeatures)
    else:
        labeled_features = extract_midi_features(dataframe)

        #Save it TODO Not working 'No such file or directory'
        #labeled_features_backup = os.path.join(args.output_folder, 'labeled_features')
        #np.save(labeled_features_backup,labeled_features)
    
    print(labeled_features)
    #Then do the job

