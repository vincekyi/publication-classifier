import sys, getopt
import string
import json
import time
import pickle
import os

import utilities.mongo as mongo
from utilities.classifier import build_and_evaluate
from random import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier



def main(argv):

    stats_db  = mongo.DBClient('mongodb://10.44.115.120:27017/pub')

    tests = 10
    iterations = 1000
    classifier = MLPClassifier
    numNodes = range(50, 300, 50)

    model_file = open('../resources/local_data.json')
    filetext = model_file.read()
    publications = json.loads(filetext)
    model_file.close()

    pubs = []
    for pub in publications:
        abstract_tokens = pub['abstract'].split('\n\n')
        abstract = pub['abstract']

        if len(abstract_tokens) >= 5:
            abstract = abstract_tokens[4]
        pub['abstract'] = abstract
        pubs.append(pub)

    X = [pub for pub in publications]
    y = ["Tool" if pub['isTool'] else "Nontool" for pub in publications]
    numSamples = len(y)
    numTools = sum([1 if pub['isTool'] else 0 for pub in publications])
    numNonTools = numSamples - numTools
    for i in numNodes:
        for j in numNodes:
            for k in numNodes:
                hidden_layers = (i,j,k)
                (model, vectorizer, stats), sec = build_and_evaluate(X,y, classifier=classifier(hidden_layer_sizes=hidden_layers), numtests=int(tests))
                stat_model = {}
                stat_model['model_type'] = 'nn'
                stat_model['test_iterations'] = tests
                stat_model['total_samples'] = numSamples
                stat_model['total_tools'] = numTools
                stat_model['total_nontools'] = numNonTools
                stat_model['accuracy'] = stats['accuracy']
                stat_model['f1'] = stats['f1']
                stat_model['precision'] = stats['precision']
                stat_model['recall'] = stats['recall']

                stat_model['avg_accuracy'] = stats['avg_accuracy']
                stat_model['avg_f1'] = stats['avg_f1']
                stat_model['avg_precision'] = stats['avg_precision']
                stat_model['avg_recall'] = stats['avg_recall']
                stats_db.insertStat(stat_model)



if __name__ == '__main__':
    main(sys.argv[1:])
