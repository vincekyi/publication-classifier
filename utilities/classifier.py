import sys, getopt
import string
import json
import time
import pickle
import os
from operator import itemgetter
import re
import utilities.urlmarker as regex

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report as clsr
import sklearn.metrics as metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split as tts
from sklearn.pipeline import FeatureUnion


from sklearn.base import BaseEstimator, TransformerMixin

def timeit(func):
    """
    Simple timing decorator
    """
    def wrapper(*args, **kwargs):
        start  = time.time()
        result = func(*args, **kwargs)
        delta  = time.time() - start
        return result, delta
    return wrapper


def identity(arg):
    """
    Simple identity function works as a passthrough.
    """
    return arg

class NLTKPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, stopwords=None, punct=None,
                 lower=True, strip=True):
        self.lower      = lower
        self.strip      = strip
        self.stopwords  = stopwords or set(sw.words('english'))
        self.punct      = punct or set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def inverse_transform(self, X):
        return [" ".join(doc) for doc in X]

    def transform(self, X):
        return [
            list(self.tokenize(doc['abstract'])) for doc in X
        ]

    def tokenize(self, document):
        # Break the document into sentences
        for sent in sent_tokenize(document):
            # Break the sentence into part of speech tagged tokens
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                # Apply preprocessing to the token
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token

                # If stopword, ignore token and continue
                if token in self.stopwords:
                    continue

                # ignore numbers
                if token.isdigit():
                    continue

                # If punctuation, ignore token and continue
                if all(char in self.punct for char in token):
                    continue

                # Lemmatize the token and yield
                lemma = self.lemmatize(token, tag)
                yield lemma

    def lemmatize(self, token, tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)


class AbstractStats(BaseEstimator, TransformerMixin):
    """Extract statistics regarding abstract concent"""

    def fit(self, x, y=None):
        return self

    def transform(self, docs):
        languages = ['python', 'java', 'html', ' r ', 'c++', ' c ', 'matlab', 'perl', 'javascript']
        repos = ['github', 'bitbucket', 'sourceforge']
        types_file = open('../resources/types.txt', 'r');
        types_text = types_file.read()
        types_tokens = types_text.split('\n')
        types_file.close()

        template_dict = {}
        for type in types_tokens:
            if not type=='':
                template_dict[type] = 0

        regex_url = re.compile(regex.URL_REGEX)
        regex_email = re.compile('[a-zA-Z0-9+_\-\.]+@[0-9a-zA-Z][.-0-9a-zA-Z]*.[a-zA-Z]+')

        result = []
        for doc in docs:
            result_dict = template_dict
            # check if the title has a colon
            result_dict['colon'] = doc['title'].find(':')

            # check the journal
            result_dict['journal name'] = doc['journal']

            # check if abstract has links
            links = regex_url.findall(doc['abstract'])
            result_dict['links'] = len(links)

            # check if links are from a code repository
            result_dict['fromRepo'] = False
            for link in links:
                for repo in repos:
                    if repo in link:
                        result_dict['fromRepo'] = True
                        break

            # check for email links
            result_dict['emails'] = len(regex_email.findall(doc['abstract']))

            # check if the abstract contains any programming languages
            result_dict['languages'] = 0
            for language in languages:
                if language in doc['abstract']:
                    result_dict['languages'] = result_dict['languages'] + 1


            # check the type of article
            for type in doc['type']:
                if type in types_tokens:
                    result_dict[type.lower()] = result_dict[type.lower()] + 1

            result.append(result_dict)

        return result


@timeit
def build_and_evaluate(X, y,
    classifier, numtests=5, outpath=None, verbose=True):

    @timeit
    def build(classifier, X, y=None, export=False):
        """
        Inner build function that builds a single model.
        """
        if isinstance(classifier, type):
            classifier = classifier()

        features = []
        tfidf = Pipeline([
            ('preprocessor', NLTKPreprocessor()),
            ('vectorizer', TfidfVectorizer(
                tokenizer=identity, preprocessor=None, lowercase=False
            ))
        ])
        features.append(('tfidf', tfidf))

        abstract = Pipeline([
                ('abstract_feature', AbstractStats()),
                ('vectorizer', DictVectorizer()),  # list of dicts -> feature matrix
            ])
        features.append(('abstract', abstract))
        feature_union = FeatureUnion(features)
        feature_extractor = Pipeline([
            ('feature_union', feature_union),
        ])
        # Label encode the targets
        X = feature_extractor.fit_transform(X)
        labels = LabelEncoder()
        y = labels.fit_transform(y)

        overall_accuracy = 0
        overall_f1 = 0
        overall_precision = 0
        overall_recall = 0

        best_accuracy = 0
        best_model = None
        best_y_test = None
        best_y_pred = None

        num_samples = 0
        stats = {}
        stats['accuracy'] = []
        stats['f1'] = []
        stats['precision'] = []
        stats['recall'] = []
        tests = numtests
        if export:
            tests = 1
        scaler = StandardScaler(with_mean=False)
        X = scaler.fit_transform(X)
        for i in range(tests):
            X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
            if classifier.__class__.__name__=='GaussianNB':
                classifier.fit(X_train.toarray(), y_train)
                y_pred = classifier.predict(X_test.toarray())
            else:
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)

            accuracy = metrics.accuracy_score(y_test, y_pred)
            f1 = metrics.f1_score(y_test, y_pred)
            precision = metrics.precision_score(y_test, y_pred)
            recall = metrics.recall_score(y_test, y_pred)

            stats['accuracy'].append(accuracy)
            stats['f1'].append(f1)
            stats['precision'].append(precision)
            stats['recall'].append(recall)


            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = classifier
                best_y_test = y_test
                best_y_pred = y_pred

        stats['avg_accuracy'] = sum(stats['accuracy'])/len(stats['accuracy'])
        stats['avg_f1'] = sum(stats['f1'])/len(stats['f1'])
        stats['avg_precision'] = sum(stats['precision'])/len(stats['precision'])
        stats['avg_recall'] = sum(stats['recall'])/len(stats['recall'])

        print('******Results after', num_samples, 'iterations*****')
        print('accuracy'.ljust(15), stats['avg_accuracy'])
        print('f1'.ljust(15), stats['avg_f1'])
        print('precision'.ljust(15), stats['avg_precision'])
        print('recall'.ljust(15), stats['avg_recall'])

        if verbose:
            print("Classification Report:\n")
            print(clsr(best_y_test, best_y_pred, target_names=labels.classes_))

        return (best_model, labels, feature_extractor, stats)


    # Begin evaluation
    if verbose: print("Building for evaluation")
    (model, labels, feature_extractor, stats), secs = build(classifier, X, y)

    if verbose:
        print("Evaluation model fit in {:0.3f} seconds".format(secs))


    if verbose:
        print("Saving best model...")
    model.labels_ = labels


    if outpath:
        with open(outpath+'model.pickle', 'wb') as f:
            pickle.dump(model, f)

        with open(outpath+'vectorizer.pickle', 'wb') as f:
            pickle.dump(feature_extractor, f)

        print("Model written out to {}".format(outpath))

    return (model, feature_extractor, stats)

def show_most_informative_features(model, vectorizer, feature, text=None, n=20):
    # Extract the vectorizer and the classifier from the pipeline
    vectorizer = vectorizer.get_params()[feature]
    classifier = model

    # Check to make sure that we can perform this computation
    if not hasattr(classifier, 'coef_'):
        raise TypeError(
            "Cannot compute most informative features on {}.".format(
                classifier.__class__.__name__
            )
        )

    if text is not None:
        # Compute the coefficients for the text
        tvec = model.transform([text]).toarray()
    else:
        # Otherwise simply use the coefficients
        if hasattr(classifier, 'coef_'):
            tvec = classifier.coef_

    # Zip the feature names with the coefs and sort
    coefs = sorted(
        zip(tvec[0], vectorizer.get_feature_names()),
        key=itemgetter(0), reverse=True
    )

    # Get the top n and bottom n coef, name pairs
    topn  = zip(coefs[:n], coefs[:-(n+1):-1])

    # Create the output string to return
    output = []

    # If text, add the predicted value to the output.
    if text is not None:
        output.append("\"{}\"".format(text))
        output.append(
            "Classified as: {}".format(model.predict([text]))
        )
        output.append("")

    # Create two columns with most negative and most positive features.
    for (cp, fnp), (cn, fnn) in topn:
        output.append(
            "{:0.4f}{: >35}    {:0.4f}{: >35}".format(
                cp, fnp, cn, fnn
            )
        )

    return "\n".join(output)

def main(argv):
    PATH = '../results/'
    modeltype = 'svm'
    modeltypes = ['svm', 'lr', 'rf', 'nn', 'nb']
    iterations = 1
    tests = 10
    try:
        opts, args = getopt.getopt(argv,"hm:i:t:",["modelname=","iter=","tests="])
    except getopt.GetoptError:
        print('classifier.py -m <modelname[svm|lr]> -i <number of iterations> -t <number of tests>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('classifier.py -m <modelname[svm|lr|rf|nn]> -i <number of iterations> -t <number of tests>')
            sys.exit()
        elif opt in ("-m", "--modelname"):
            if arg in modeltypes:
                modeltype = arg
        elif opt in ("-i", "--numIter"):
            iterations = int(arg)
        elif opt in ("-t", "--tests"):
            tests = arg

    print('Model:', modeltype,', iterations:', iterations, ', test iterations:', tests)
    if not os.path.exists(PATH+modeltype+'_model.pickle'):
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

        if modeltype=='lr':
            classifier = LogisticRegression(max_iter=int(iterations))
        elif modeltype=='rf':
            classifier = RandomForestClassifier()
        elif modeltype=='nn':
            classifier = MLPClassifier(max_iter=iterations, hidden_layer_sizes=(100,100,100))
        elif modeltype=='nb':
            classifier = GaussianNB()
        else:
            classifier = SGDClassifier(n_iter=iterations)
        (model, vectorizer, stats), sec = build_and_evaluate(X,y, classifier=classifier, numtests=int(tests), outpath=PATH+modeltype+'_')

    else:
        model_file = open(PATH+modeltype+'_model.pickle', 'rb')
        vectorizer_file = open(PATH+modeltype+'_vectorizer.pickle', 'rb')
        model = pickle.load(model_file)
        vectorizer = pickle.load(vectorizer_file)
        model_file.close()
        vectorizer_file.close()
        print(show_most_informative_features(model, vectorizer, 'feature_union__tfidf__vectorizer'))
        print('*******')
        print(show_most_informative_features(model, vectorizer, 'feature_union__abstract__vectorizer', n=50))

if __name__ == '__main__':
    main(sys.argv[1:])
