from abc import ABCMeta, abstractmethod
import utilities.mongo as mongo
import json
import nltk
import string
import re, urllib
import utilities.urlmarker as regex
from nltk.corpus import stopwords
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np



class FeatureExtraction:
    def __init__(self, fromLocal, features):
        if fromLocal:
            with open('./resources/local_data.json') as file:
                filetext = file.read()
                self.publications = json.loads(filetext)
        else:
            db  = mongo.DBClient('mongodb://10.44.115.120:27017/pub')
            self.publications = db.queryAll()
        self.numPubs = len(self.publications)

        basic_stats = {}
        basic_stats['tools'] = 0
        basic_stats['nontools'] = 0
        for pub in self.publications:
            if pub['isTool']:
                basic_stats['tools'] = basic_stats['tools'] + 1
            else:
                basic_stats['nontools'] = basic_stats['nontools'] + 1

        basic_stats['total'] = len(self.publications)

        self.feature_obj = []
        self.words = {}
        self.sentences = {}
        tokenizeWords = False
        tokenizeSent = False
        for feature in features:
            new_feature = feature(basic_stats)
            tokenizeWords = tokenizeWords or new_feature.requireWordTokenized
            tokenizeSent = tokenizeSent or new_feature.requireSentTokenized

            self.feature_obj.append(new_feature)

        for i in range(self.numPubs):
            pub = self.publications[i]
            abstract_tokens = pub['abstract'].split('\n\n')
            abstract = pub['abstract']

            if len(abstract_tokens) >= 5:
                abstract = abstract_tokens[4]
            self.publications[i]['abstract'] = abstract

            if tokenizeWords:
                word_tokens = [a.lower() for a in nltk.word_tokenize(abstract)]
                remove_stop = [w for w in word_tokens if not w in stopwords.words('english')]
                translator = str.maketrans({key: None for key in string.punctuation})
                no_punct = [w.translate(translator) for w in remove_stop]
                self.words[pub['pmid']] = no_punct

            if tokenizeSent:
                remove_newlines = abstract.replace('\n', ' ')
                abstract_sentences = nltk.sent_tokenize(remove_newlines)
                self.sentences[pub['pmid']] = abstract_sentences



    def getStats(self):
        for pub in self.publications:
            for feature in self.feature_obj:
                if not feature.analyzeWholeCorpus:
                    if feature.requireWordTokenized:
                        pub['words'] = self.words[pub['pmid']]
                    if feature.requireSentTokenized:
                        pub['sentences'] = self.sentences[pub['pmid']]
                    feature.calculateStats(pub)

        for feature in self.feature_obj:
            if feature.analyzeWholeCorpus:
                feature.calculateStats(self.publications)

        for feature in self.feature_obj:
            feature.printStats()



class Feature(metaclass=ABCMeta):
    statName = "Feature"
    examples = []
    def __init__(self, basic_stats):
        self.numPubs = basic_stats['total']
        self.numTools = basic_stats['tools']
        self.numNonTools = basic_stats['nontools']
        self.requireWordTokenized = False
        self.requireSentTokenized = False
        self.analyzeWholeCorpus = False
        self.result_dict = {}

    @abstractmethod
    def calculateStats(self, pub):
        raise NotImplementedError()

    def printStats(self):
        print("***************", self.statName,"***************")



class ColonFeature(Feature):

    def __init__(self, basic_stats):
        super().__init__(basic_stats)
        self.statName = "Colon in title"
        self.result_dict = {}
        self.colon_stats = {}
        self.colon_stats['count'] = 0
        self.colon_stats['tools'] = []
        self.colon_stats['nontools'] = []
        self.colon_stats['tool_examples'] = []
        self.colon_stats['nontool_examples'] = []

    def calculateStats(self, pub):
        if pub['title'].find(':') > 0:

            self.colon_stats['count'] = self.colon_stats['count'] + 1
            if pub['isTool']:
                self.colon_stats['tools'].append(pub['pmid'])
                self.colon_stats['tool_examples'].append(pub['title'])
            else:
                self.colon_stats['nontools'].append(pub['pmid'])
                self.colon_stats['nontool_examples'].append(pub['title'])

        self.result_dict[':'] = self.colon_stats

        self.result_dict[':'] = self.colon_stats
        return self.result_dict

    def printStats(self):
        super().printStats()
        print('Colon in tools:'.ljust(50)+str(len(self.colon_stats['tools']))+'/'+str(self.numTools))
        print('Colon in nontools:'.ljust(50)+str(len(self.colon_stats['nontools']))+'/'+str(self.numNonTools))




class LanguageFeature(Feature):
    def __init__(self, basic_stats):
        super().__init__(basic_stats)
        self.statName = "Programming Language in abstract"
        self.result_dict = {}

        with open('./resources/languages.txt', 'r') as file:
            filetext = file.read()

            self.languages = [l.lower() for l in filetext.split('\n')]
            self.requireWordTokenized = True


    def calculateStats(self, pub):
        for language in self.languages:
            abstract_tokens = pub['words']
            if language in abstract_tokens:
                if language in self.result_dict:
                    stats = self.result_dict[language]
                    stats['count'] = stats['count'] + 1
                    if pub['isTool']:
                        stats['tools'].append(pub['pmid'])
                    else:
                        stats['nontools'].append(pub['pmid'])
                    self.result_dict[language] = stats
                else:
                    stats = {}
                    stats['count'] = 1
                    if pub['isTool']:
                        stats['tools'] = [pub['pmid']]
                        stats['nontools'] = []
                    else:
                        stats['nontools'] = [pub['pmid']]
                        stats['tools'] = []
                    self.result_dict[language] = stats

        return self.result_dict

    def printStats(self):
        super().printStats()

        for language in sorted(self.result_dict.items(), key=lambda element:element[1]['count']):
            print(language[0].ljust(50)+str(language[1]['count']))
            print('\tTools:'.ljust(30)+str(len(language[1]['tools']))+'/'+str(self.numTools))
            print('\tNon Tools:'.ljust(30)+str(len(language[1]['nontools']))+'/'+str(self.numNonTools))



class PublicationType(Feature):
    def __init__(self, basic_stats):
        super().__init__(basic_stats)
        self.statName = "Type of publication"
        self.result_dict = {}

    def calculateStats(self, pub):
        for type in pub['type']:
            lowercase_type = type.lower().strip()
            if lowercase_type in self.result_dict:
                stats = self.result_dict[lowercase_type]
                stats['count'] = stats['count'] + 1
                if pub['isTool']:
                    stats['tools'].append(pub['pmid'])
                else:
                    stats['nontools'].append(pub['pmid'])
                self.result_dict[lowercase_type] = stats

            else:
                stats = {}
                stats['count'] = 1
                if pub['isTool']:
                    stats['tools'] = [pub['pmid']]
                    stats['nontools'] = []
                else:
                    stats['nontools'] = [pub['pmid']]
                    stats['tools'] = []
                self.result_dict[lowercase_type] = stats
        return self.result_dict

    def printStats(self):
        super().printStats()
        for type in sorted(self.result_dict.items(), key=lambda element:element[1]['count']):
            print(type[0].ljust(50)+str(type[1]['count']))
            print('\tTools:'.ljust(30)+str(len(type[1]['tools']))+'/'+str(self.numTools))
            print('\tNontools:'.ljust(30)+str(len(type[1]['nontools']))+'/'+str(self.numNonTools))



class JournalType(Feature):
    def __init__(self, basic_stats):
        super().__init__(basic_stats)
        self.statName = "Type of journal"
        self.result_dict = {}


    def calculateStats(self, pub):
        lowercase_journal = pub['journal'].lower().strip()
        if lowercase_journal in self.result_dict:
            stats = self.result_dict[lowercase_journal]
            stats['count'] = stats['count'] + 1
            if pub['isTool']:
                stats['tools'].append(pub['pmid'])
            else:
                stats['nontools'].append(pub['pmid'])
            self.result_dict[lowercase_journal] = stats
        else:
            stats = {}
            stats['count'] = 1
            if pub['isTool']:
                stats['tools'] = [pub['pmid']]
                stats['nontools'] = []
            else:
                stats['nontools'] = [pub['pmid']]
                stats['tools'] = []
            self.result_dict[lowercase_journal] = stats

        return self.result_dict

    def printStats(self):
        super().printStats()

        for journal in sorted(self.result_dict.items(), key=lambda element:element[1]['count']):
            print(journal[0].ljust(50)+str(journal[1]['count']))
            print('\tTools:'.ljust(30)+str(len(journal[1]['tools']))+'/'+str(self.numTools))
            print('\tNontools:'.ljust(30)+str(len(journal[1]['nontools']))+'/'+str(self.numNonTools))

class ViewedPaper(Feature):
    def __init__(self, basic_stats):
        super().__init__(basic_stats)
        self.statName = "Percentage of Publications that were Determined from Paper"
        self.result_dict['paperViewed'] = {'tools_viewed': 0,
                                            'nontools_viewed': 0}


    def calculateStats(self, pub):
        tools_viewed = self.result_dict['paperViewed']['tools_viewed']
        nontools_viewed = self.result_dict['paperViewed']['nontools_viewed']
        if pub['fullTextViewed']:
            if pub['isTool']:
                tools_viewed = tools_viewed + 1
            else:
                nontools_viewed = nontools_viewed + 1

        self.result_dict['paperViewed'] = {'tools_viewed': tools_viewed,
                                            'nontools_viewed': nontools_viewed}
        return self.result_dict

    def printStats(self):
        super().printStats()

        results = self.result_dict['paperViewed']
        total = results['tools_viewed']+results['nontools_viewed']
        print('Total viewed:'.ljust(50), total, total*100/self.numPubs, '%')
        print('\tTools:'.ljust(30),results['tools_viewed'],'/',self.numTools)
        print('\tNontools:'.ljust(30),results['nontools_viewed'],'/',self.numNonTools)

class SentenceURL(Feature):
    def __init__(self, basic_stats):
        super().__init__(basic_stats)
        self.statName = "Sentences with URL in abstract"
        self.requireSentTokenized = True
        self.result_dict = {}
        self.url_stats = {}
        self.url_stats['count'] = 0
        self.url_stats['tools'] = []
        self.url_stats['nontools'] = []
        self.url_stats['tool_examples'] = []
        self.url_stats['nontool_examples'] = []


    def calculateStats(self, pub):
        regex_email = re.compile('[a-zA-Z0-9+_\-\.]+@[0-9a-zA-Z][.-0-9a-zA-Z]*.[a-zA-Z]+')
        regex_url = re.compile(regex.URL_REGEX)
        hasURL = False
        examples = []
        for sentence in pub['sentences']:
            if len(regex_url.findall(sentence)) > 0 and len(regex_email.findall(sentence))==0:
                hasURL = True
                examples.append(sentence)

        if hasURL:
            self.url_stats['count'] = self.url_stats['count'] + 1
            if pub['isTool']:
                self.url_stats['tools'].append(pub['pmid'])
                self.url_stats['tool_examples'].append(examples)
            else:
                self.url_stats['nontools'].append(pub['pmid'])
                self.url_stats['nontool_examples'].append(examples)

        self.result_dict['url'] = self.url_stats

        return self.result_dict

    def printStats(self):
        super().printStats()
        stats = self.result_dict['url']
        print('Total publications with URLs'.ljust(50)+str(stats['count']))
        print('\tTools:'.ljust(30)+str(len(stats['tools']))+'/'+str(self.numTools))
        print('\tNontools:'.ljust(30)+str(len(stats['nontools']))+'/'+str(self.numNonTools))
        print('***Tools')
        for sent in stats['tool_examples']: print(sent)
        print('***Nontools')
        for sent in stats['nontool_examples']: print(sent)

class TermFeature(Feature):
    def __init__(self, basic_stats):
        super().__init__(basic_stats)
        self.statName = "Most frequent terms"
        self.result_dict = {}
        self.requireWordTokenized = True
        self.stemmer = PorterStemmer()


    def calculateStats(self, pub):
        for word in pub['words']:
            stemmed_word = self.stemmer.stem(word)
            if stemmed_word in self.result_dict:
                stats = self.result_dict[stemmed_word]
                stats['count'] = stats['count'] + 1
                if pub['isTool']:
                    stats['tools'].append(pub['pmid'])
                else:
                    stats['nontools'].append(pub['pmid'])
                self.result_dict[stemmed_word] = stats
            else:
                stats = {}
                stats['count'] = 1
                if pub['isTool']:
                    stats['tools'] = [pub['pmid']]
                    stats['nontools'] = []
                else:
                    stats['nontools'] = [pub['pmid']]
                    stats['tools'] = []
                self.result_dict[stemmed_word] = stats

        return self.result_dict

    def printStats(self):
        super().printStats()

        for term in sorted(self.result_dict.items(), key=lambda element:element[1]['count']):
            numTools = len(term[1]['tools'])
            numNonTools = len(term[1]['nontools'])
            if term[1]['count'] > 5:
                significant = ''
                if (numTools > 1.5*numNonTools or numNonTools > 1.5*numTools):
                    significant = '**'
                print(term[0].ljust(50)+str(term[1]['count'])+significant)
                print('\tTools:'.ljust(30)+str(len(term[1]['tools']))+'/'+str(self.numNonTools))
                print('\tNontools:'.ljust(30)+str(len(term[1]['nontools']))+'/'+str(self.numNonTools))

class TFIDFFeature(Feature):
    def __init__(self, basic_stats):
        super().__init__(basic_stats)
        self.statName = "TFIDF of abstracts"
        self.result_dict = {}
        self.analyzeWholeCorpus = True

    def tokenize(self, text):
        stemmer = PorterStemmer()
        tokens = nltk.word_tokenize(text)
        translator = str.maketrans({key: None for key in string.punctuation})
        no_punct = [w.translate(translator) for w in tokens]
        stemmed = []
        for item in no_punct:
            stemmed.append(stemmer.stem(item))

        return stemmed

    def calculateStats(self, pub):
        overall_dict = {}
        tool_dict = {}
        nontool_dict = {}
        for p in pub:
            if p["isTool"]:
                tool_dict[p['pmid']] = p['abstract']
            else:
                nontool_dict[p['pmid']] = p['abstract']
            overall_dict[p['pmid']] = p['abstract']

        overall_tfidf = TfidfVectorizer(tokenizer=self.tokenize, stop_words='english')
        overall = overall_tfidf.fit_transform(overall_dict.values())

        tool_tfidf = TfidfVectorizer(tokenizer=self.tokenize, stop_words='english')
        tools = tool_tfidf.fit_transform(tool_dict.values())

        nontool_tfidf = TfidfVectorizer(tokenizer=self.tokenize, stop_words='english')
        nontools = nontool_tfidf.fit_transform(nontool_dict.values())


        self.result_dict['overall_words'] = overall_tfidf.get_feature_names()
        self.result_dict['overall_matrix'] = overall.todense()

        self.result_dict['tool_words'] = tool_tfidf.get_feature_names()
        self.result_dict['tool_matrix'] = tools.todense()

        self.result_dict['nontool_words'] = nontool_tfidf.get_feature_names()
        self.result_dict['nontool_matrix'] = nontools.todense()

        return self.result_dict

    def printStats(self):
        super().printStats()

        word_dict = {}
        types = ['overall', 'tool', 'nontool']
        for type in types:
            result = np.zeros(len(self.result_dict[type+'_words']))
            for row in self.result_dict[type+'_matrix']:
                result = result + row

            result_list = result.tolist()[0]
            word_dict[type] = {}
            for i in range(len(self.result_dict[type+'_words'])):
                word = self.result_dict[type+'_words'][i]
                word_dict[type][word] = result_list[i]

        sorted_results = sorted(word_dict['tool'].items(), key=lambda element:element[1])

        print('Word'.ljust(20), 'Tool'.ljust(20), 'Nontool'.ljust(20), 'Overall'.ljust(20))
        for item in sorted_results[-200:]:
            print(item[0].ljust(20), str(item[1]).ljust(20), str(word_dict['nontool'][item[0]]).ljust(20), str(word_dict['overall'][item[0]]).ljust(20))




if __name__ == '__main__':
    fe = FeatureExtraction(True, [ViewedPaper])
    fe.getStats()
