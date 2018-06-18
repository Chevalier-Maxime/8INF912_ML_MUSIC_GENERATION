import sys
assert sys.version_info < (3, 0)

from sklearn.externals import joblib
import select_features as sf
import extract_features as ef
from argparse import ArgumentParser
import os
import random
import numpy as np
from sklearn import tree

import plot_unveil_tree_structure as pretty_path
#from sklearn.ensemble import RandomForestClassifier

#Might not work due to OS privileges
def exctract_features_from_midi_files(input_folder):
    print(os.path.join(input_folder,'midi'))
    print(not os.path.isdir(os.path.join(input_folder,'midi')))
    if not os.path.isdir(os.path.join(input_folder,'GENERATED','midi')):
        os.mkdir(os.path.join(input_folder,'GENERATED'))
        os.mkdir(os.path.join(input_folder,'GENERATED','midi'))
        #Moves files
        source = input_folder
        dest = os.path.join(source, 'GENERATED','midi')
        files = os.listdir(source)
        for f in files:
            for retry in range(100):
                try:
                    os.rename(os.path.join(source,f), os.path.join(dest,f))
                    break
                except:
                    print "rename failed, retrying..."
           
    if not os.path.isdir(os.path.join(input_folder,'GENERATED','features')):
        os.mkdir(os.path.join(input_folder,'GENERATED','features'))
    
    ef.main(input_folder)

def test_features_from_midi_files(args):
    model_filename = 'finalized_model_random_Forest.pkl'
    model_path_filename = os.path.join(args.input_model, model_filename)

    features_filename = 'finalized_features_random_Forest.pkl'
    features_path_filename = os.path.join(args.input_model, features_filename)

    excluded_filename = 'finalized_excluded_random_Forest.pkl'
    excluded_path_filename = os.path.join(args.input_model, excluded_filename)


    model = joblib.load(model_path_filename)
    features_index = joblib.load(features_path_filename)
    features_index.sort()
    excluded_features = joblib.load(excluded_path_filename)

    vecs = sf.build_vectors(excluded_features,args.input_folder)

    # Not needed random.shuffle(vecs)
    X, _ = map(lambda x: x[0:len(x)-2], vecs), map(lambda x: x[len(x)-2], vecs)
    

    # print(vecs[0])
    # print(len(vecs[0]))

    # print(X[0])
    # print(features_index)
    X_new = []
    for full_features in X:
        feature_tmp = []
        for i in features_index:
            feature_tmp.append(full_features[i])
        X_new.append(feature_tmp)

    # print(X_new[0])
    # print(X_new)
    # print(len(X_new[0]))
    #print(X)
    X_array = np.asarray(X_new)
    y_predicted = model.predict_proba(X_array)
    y_predicted_Label = model.predict(X_array)

    # print(len(y_predicted))
    # print(X_array[:1])
    #print("file : %s is classified as : %s" % (vecs[0][len(vecs[0])-1],y_predicted[0]))

    for i in range(0,len(y_predicted)):
        print("file : %s is classified as : %s : %s" % (vecs[i][len(vecs[i])-1],y_predicted[i],y_predicted_Label[i]))
    
    #TODO Debug option
    # pretty_path.pretty_path(model.estimators_[0])
    # dotfile = open("dt.dot", 'w')
    # tree.export_graphviz(model.estimators_[0], out_file=dotfile)
    # dotfile.close()
            


    
    # print(y_predicted)

#TODO test if features exists before lunching
if __name__ == "__main__":
    parser = ArgumentParser()

    # Mandatory arguments
    parser.add_argument('input_folder', type=str, help='The folder containing \
            the midi files to test')
    parser.add_argument('output_folder', type=str, help='The folder where the \
            midi files will be copied by class')
    parser.add_argument('input_model', type=str, help='The folder which contain \
            model\'s files')

    # Optional argument for feature extraction
    parser.add_argument('--feature_extraction', action="store_true", help='Does the program\
            need to extract features from midi files (default FALSE)')
    
    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        print('%s is not a directory' % args.input_folder)
        sys.exit()
    
    if not os.path.isdir(args.input_model):
        print('%s is not a directory' % args.input_folder)
        sys.exit()

    if args.feature_extraction:
        exctract_features_from_midi_files(args.input_folder)

    test_features_from_midi_files(args)



