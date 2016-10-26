import sys, getopt
import string
import json
import time
import pickle
import os

from utilities.classifier import build_and_evaluate
from random import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

def plot(title, xlabel, ylabel, data):
    plt.figure()
    plt.title(title)
    plt.ylim(0, 2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    for line in data:
        plotline(line['label'], line['x'], line[title+'_y'])

    plt.legend(loc="best")

def plotline(label, xvalues, yvalues):
    plt.plot(xvalues, yvalues, 'o-', label=label)

def main(argv):
    PATH = '../results/'
    tests = 10
    classifier = MLPClassifier
    layers = range(1,2)


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


    plot_data = []
    i = 2
    data = {}
    data['x'] = [j for j in range(100, 400, 100)]
    data['label'] = 'Layers: '+str(i)
    data['accuracy_y'] = []
    data['precision_y'] = []
    data['recall_y'] = []
    data['f1_y'] = []
    # for second in data['x']:
    #     hidden_layers = (300, second)
    #     print(hidden_layers)
    #     (model, vectorizer, stats), sec = build_and_evaluate(X,y, classifier=classifier(hidden_layer_sizes=hidden_layers), numtests=int(tests))
    #     data['accuracy_y'].append(stats['avg_accuracy'])
    #     data['precision_y'].append(stats['avg_precision'])
    #     data['recall_y'].append(stats['avg_recall'])
    #     data['f1_y'].append(stats['avg_f1'])
    # plot_data.append(data)
    # plot('accuracy', 'x', 'y', plot_data)
    # plot('f1', 'x', 'y', plot_data)
    # plot('precision', 'x', 'y', plot_data)
    # plot('recall', 'x', 'y', plot_data)
    #
    # plt.show()

    # (model, vectorizer, stats), sec = build_and_evaluate(X,y, classifier=classifier(hidden_layer_sizes=(100,200,100)), numtests=20)
    (model, vectorizer, stats), sec = build_and_evaluate(X,y, classifier=classifier(hidden_layer_sizes=(100,200,200)), numtests=20)
    # (model, vectorizer, stats), sec = build_and_evaluate(X,y, classifier=classifier(hidden_layer_sizes=(200,200,200)), numtests=20)
    # (model, vectorizer, stats), sec = build_and_evaluate(X,y, classifier=classifier(hidden_layer_sizes=(300,200,200)), numtests=20)
    # (model, vectorizer, stats), sec = build_and_evaluate(X,y, classifier=classifier(hidden_layer_sizes=(300,300,300)), numtests=20)





if __name__ == '__main__':
    main(sys.argv[1:])
