# REQUIRE PYTHON v2
# coding: utf-8
#Thanks to https://github.com/cjnolet/midi_genre_corpus for midi features
#Thanks to https://stackoverflow.com/questions/492519/timeout-on-a-function-call/494273 for timeout on functions
from __future__ import print_function
import sys
import threading
from time import sleep
try:
    import thread
except ImportError:
    import _thread as thread

import sys
assert sys.version_info < (3, 0)

from argparse import ArgumentParser
from music21 import converter
from music21 import features
from music21 import environment
import threading
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed




#Thanks to https://stackoverflow.com/questions/492519/timeout-on-a-function-call/494273 (suite)
try:
    range, _print = xrange, print
    def print(*args, **kwargs): 
        flush = kwargs.pop('flush', False)
        _print(*args, **kwargs)
        if flush:
            kwargs.get('file', sys.stdout).flush()            
except NameError:
    pass


def quit_function(fn_name):
    # print to stderr, unbuffered in Python 2.
    print('{0} took too long'.format(fn_name), file=sys.stderr)
    sys.stderr.flush() # Python 3 stderr is likely buffered.
    thread.interrupt_main() # raises KeyboardInterrupt


def exit_after(s):
    '''
    use as decorator to exit process if 
    function takes longer than s seconds
    '''
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer


def extracted_features(file_path):
    ret = []
    if os.path.exists(file_path):
        with open(file_path) as f:
            content = f.readlines()
            if len(content) > 0:
                try:
                    ret = map(lambda it: it.split(",")[0:2], content)
                    ret = map(lambda it: (it[0], it[1]), ret)
                except:
                    print("Error in resume.")
        f.close()
    return set(ret)


@exit_after(120)
def extract_vector(feature):
    return feature.extract().vector


def extract(path):
    print("Opening coverter for %s" % path)
    try:
        o = converter.parse(path)
        #TODO change this aberation ! URGENT
        features_path = os.path.join(path,"..","..","features", os.path.basename(path) + ".csv")

        # Allow the continuation of extraction if, for some reason, an error occurred
        already_extracted = extracted_features(features_path)

        fs = features.jSymbolic.extractorsById
        for k in fs:
            if k is not "I":
                for i in range(len(fs[k])):
                    if (k, str(i)) not in already_extracted:
                        if fs[k][i] is not None:
                            n = fs[k][i].__name__
                        if fs[k][i] in features.jSymbolic.featureExtractors:
                            print("Extracting " + str(n) + " from " + features_path)
                            t5 = features.jSymbolic.getExtractorByTypeAndNumber(k, i)(o)
                            try:
                                vec = extract_vector(t5)
                                text_file = open(features_path, "a")
                                text_file.write(k + "," + str(i) + "," + str(n) + "," + str(list(vec)).replace(' ', '').replace('[', '').replace(']', '') + "\n")
                                text_file.flush()
                                text_file.close()
                            except (KeyboardInterrupt, Exception) as e:
                                print(e)
                                print("Error extracting " + str(n) + " from " + features_path + " continuing...")
    except Exception as e:
        print(e)
        print("Failure encountered extracting features from %s" % path )

    print("Finished processing %s" % path)


def main(input_folder):
    #Get all genres
    basedir = input_folder
    correct_form = True
    for dir in next(os.walk(basedir))[1]:
        genres.append(dir)
        #Verify architecture of subdirectory
        if not os.path.isdir(os.path.join(basedir,dir,'midi')):
            print('Folder %s must contain a subfolder \'midi\' with all files')
            correct_form = False
        if not os.path.isdir(os.path.join(basedir,dir,'features')):
            os.makedirs(os.path.join(basedir,dir,'features'))
    
    if not correct_form :
        print('Please resolves the errors')
        sys.exit()

    final_mids = []
    for g in genres:
        print(g)
        mids = os.listdir(os.path.join(basedir, g, "midi"))
        for i in mids:
            print(os.path.join(basedir, g, "midi",i))
            if(i.endswith("midi") or i.endswith("mid")):
                final_mids.append(os.path.join(basedir, g, "midi",i))
    
    process_args = (path_midi_file for path_midi_file in final_mids)
    with ProcessPoolExecutor() as executor:
        for result in executor.map(extract, process_args):
            pass


    print("Done extracting features.")

#Music21 warn about python 2.7
environment.UserSettings()['warnings'] = 0


basedir = ''
genres = []

#TODO Add a progress bar (like preprocess.py)
if __name__ == "__main__":
    parser = ArgumentParser()

    # Mandatory arguments
    parser.add_argument('input_folder', type=str, help='The folder containing \
            the game folders containing genre folder which contains midi files (*.mid or *.midi)')
    
    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        print('%s is not a directory' % args.input_folder)
        sys.exit()
    
    main(args.input_folder)

