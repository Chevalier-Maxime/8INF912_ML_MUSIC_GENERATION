# coding: utf-8

#Thanks to https://github.com/cjnolet/midi_genre_corpus

import sys

from argparse import ArgumentParser

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold
from sklearn.externals import joblib

import os
import sys
import string
from music21 import features
from itertools import chain
import csv
import json
import random
import itertools

#TODO return nested should be enought ? :/
def fetch_genres(basedir):
    genres = []
    nested = os.listdir(basedir)
    for i in nested:
        try:
            if(os.path.isdir(os.path.join(basedir,i)) and not i.startswith(".")):
                genres.append(i)
        except:
            print("An error occured trying to load features for genre " + i)
    return genres


def flatmap(l):
    return [item for sublist in l for item in sublist]


def failed_features(basedir):
    not_included = set()
    fs = features.jSymbolic.extractorsById
    genres = fetch_genres(basedir)
    for genre in genres:
        try: 
            filename = os.path.join(basedir , genre ,"features")
            files = os.listdir(filename)
            for aFile in files:
                if aFile.endswith(".csv"):
                    arr = fileToArray(os.path.join(filename,aFile))
                    arr1 = set(map(lambda x: tuple(x[0:2]), arr))
                    # If features failed to extract 
                    for k in fs:
                        if k is not "I":
                            for i in range(len(fs[k])):
                                if (k,str(i)) not in arr1 and fs[k][i] is not None:
                                    not_included.add((k,i))
        except:
            print("An error occured trying to load features for genre " + genre)
    return not_included


def build_vectors(exclude_features, basedir):
    genres = fetch_genres(basedir)
    final_vecs = []
    for genre in genres:
        try:
            filename = os.path.join(basedir , genre ,"features")
            files = os.listdir(filename)
            for aFile in files:
                vec = []
                if aFile.endswith(".csv"):
                    arr = fileToArray(os.path.join(filename,aFile))
                    for i in arr:
                        if (i[0], int(i[1])) not in exclude_features:
                            vec.append(map(lambda x: float(x), i[3:]))
                if len(vec) > 0:
                    final_vec = flatmap(vec)
                    final_vec.append(genre)
                    final_vec.append(aFile)
                    final_vecs.append(final_vec)
                    
        except:
            print("Error occured trying to load features for " + genre)
    return final_vecs


def fileToArray(filename):
    array = []
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        try:
            array = list(reader)
        except: 
            print("Error reading: " + filename)
    f.close()
    return array

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = np.asarray(cm)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def read_data(basedir):

    # Features that failed to extract will not be written to files. Put all the failed
    # features into a set so that they can be excluded from the final vectors right away.
    not_included = failed_features(basedir)

    # Vectors is a list of lists. The inner list contains the features and the resulting
    # labels in the last position of each vector. 
    vecs = build_vectors(not_included, basedir)
    
    return (vecs, not_included)


def select_features(X, y):

    clf = ExtraTreesClassifier()
    clf = clf.fit(X, y)
    importances = clf.feature_importances_ 
    model = SelectFromModel(clf, prefit=True)
    
    X_new = model.transform(X)

    ranks = []
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    for f in range(X_new.shape[1]):
        ranks.append((f + 1, indices[f], importances[indices[f]]))

    return (X_new, ranks)

def main_RandomForest(basedir):
    genres = fetch_genres(basedir)
    print(genres)

    matrices = []
    accuracies = []

    sel_features = []
    
    best_model = None
    best_Accuracy = 0
    best_cm = None
    best_features = None
    best_excluded = None


    for i in range(0,5):
        vecs, excluded = read_data(basedir)
        random.shuffle(vecs)
        
        X, y = map(lambda x: x[0:len(x)-2], vecs), map(lambda x: x[len(x)-2], vecs)
        
        selected = select_features(X, y)
        X_new = selected[0]
        
        sel_features.append(set(map(lambda x: x[1], selected[1])))
        
        print("Feature Rankings for run " + str(i))
        # for rank in selected[1]:
        #     print(rank)
        # print(np.asarray(X[0]))
        # print(X_new[0])
        print("Number of features : %d" % len(selected[1]))
        
        X_fin = np.array(X_new)
        Y_fin = np.array(y)
        
        kf = StratifiedKFold(Y_fin, 2)

        count = 0
        for train_index, test_index in kf:
            mod = RandomForestClassifier(n_estimators=100, random_state=0, criterion="entropy")

            clf = mod.fit(X_fin[train_index], Y_fin[train_index])

            X_test = X_fin[test_index]
            Y_test = Y_fin[test_index]

            Y_pred = clf.predict(X_test)

            cm = confusion_matrix(Y_test, Y_pred, genres)
            #plot_confusion_matrix(cm, genres, normalize=True)
            cr = classification_report(Y_test, Y_pred)
            
            #print(str(cr))
            
            a_score = accuracy_score(Y_test, Y_pred, True)
            if a_score > best_Accuracy :
                best_Accuracy = a_score
                best_model = clf
                best_cm = cm
                best_features = selected[1]
                best_excluded = excluded
            matrices.append(cm)
            accuracies.append(a_score)
        
    # Average together the confusion matrix values
    final_cf = []
    for i in range(0, len(matrices[0])):
        final_cf.append([])
        for j in range(0, len(matrices[0])):

            sum = 0
            for k in range(0, len(matrices)):
                sum += matrices[k][i][j]
            final_cf[i].append(sum / len(matrices))

    sum = 0
    for i in accuracies:
        sum += i

    # plot_confusion_matrix(final_cf, genres, normalize=True)
    # plt.show()

    # plt.clf()

    plot_confusion_matrix(best_cm, genres, normalize=True)
    #plt.show()

    
    print(str(sum / len(accuracies)))

    print(str(len(sel_features[0])))
    print(str(len(set.intersection(*sel_features))))

    # print(sel_features)

    return (best_model, best_features, best_excluded)



if __name__ == "__main__":
    parser = ArgumentParser()

    # Mandatory arguments
    parser.add_argument('input_folder', type=str, help='The folder containing \
            the game folders containing genre folder which contains features folders (*.csv)')
    parser.add_argument('output_folder', type=str, help='The folder where the \
            model will be saved')
    
    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        print('%s is not a directory' % args.input_folder)
        sys.exit()
    
    #Get all genres
    basedir = args.input_folder

    model, features, excluded = main_RandomForest(basedir)

    features_indice = []
    for rank in features:
        features_indice.append(rank[1])

    if not os.path.isdir(args.output_folder):
        os.mkdir(args.output_folder)
    
    model_filename = 'finalized_model_random_Forest.pkl'
    model_path_filename = os.path.join(args.output_folder, model_filename)

    features_filename = 'finalized_features_random_Forest.pkl'
    features_path_filename = os.path.join(args.output_folder, features_filename)

    excluded_filename = 'finalized_excluded_random_Forest.pkl'
    excluded_path_filename = os.path.join(args.output_folder, excluded_filename)

    #if contain one, assume it contain all
    if 'finalized_model_random_Forest.pkl' in os.listdir(args.output_folder):
        try:
            os.remove(model_path_filename)
            os.remove(features_path_filename)
            os.remove(excluded_path_filename)
        except:
            pass

    #model.write().overwrite().save(path_filename)
    joblib.dump(model, model_path_filename)
    joblib.dump(features_indice, features_path_filename)
    joblib.dump(excluded, excluded_path_filename)
    print('Model succefully saved into %s' % model_path_filename)

    