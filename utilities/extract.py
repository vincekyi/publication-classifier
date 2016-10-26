# extract.py
# author: Allison Ko

#####################################################################
# INPUT: data.csv with pubmed IDs and 1/0 tool or not tool, e.g:
# 1234567,1
# 9876543,0
#
# OUTPUT: json object including pmid, text from title and abstract, e.g.:
# [{pmid:<int>, is_tool:<true,false>, title:<string>, abstract:<string>},...]
#####################################################################

# Dependencies
import requests
import csv
import xml.etree.ElementTree as ET
import re
import sys
import json
import argparse
import threading
from bs4 import BeautifulSoup
from time import sleep


PUBMED_RETMAX = 1000
PUBMED_ID_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&rettype=xml&id='
PUBMED_ABSTRACT_URL = 'https://www.ncbi.nlm.nih.gov/pubmed/{0}?report=abstract&format=text'
PUBMED_JOURNAL_URL = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&retmax='+str(PUBMED_RETMAX)+'&sort=relevance&field=journal&term="{0}"&retstart={1}'
DOI_ID_URL = 'http://dx.doi.org/'
DOI_REGEX = '\b(10[.][0-9]{4,}(?:[.][0-9]+)*/(?:(?!["&\'<>])[[:graph:]])+)\b'

def extractFromJournal(journal, numThreads=1):
    pubIDs = [];
    moreResults = True
    start = 0
    while moreResults:
        tempURL = PUBMED_JOURNAL_URL.format(str(journal), start)
        req = requests.get(tempURL)
        obj = json.loads(req.text)
        pubIDs = pubIDs+ obj['esearchresult']['idlist']

        if int(obj['esearchresult']['retstart'])+PUBMED_RETMAX > int(obj['esearchresult']['count']):
            moreResults = False
        else:
            start = start + PUBMED_RETMAX
    return retrievePub(pubIDs, numThreads)



def extractFromFile(input='data.csv', numThreads=1, output=None):
    # read data from .csv
    pubIDs = []
    pub_dict = dict() # keeps track of [pmid, obj(id, title, abstract, is_tool)] values

    # parse values from input file and store in dictionary
    with open(input) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            pubIDs.append(row[0])
            obj = {}
            obj['pmid'] = int(row[0])
            obj['is_tool'] = (True if int(row[1]) == 1 else False)
            pub_dict[row[0]] = obj

    results = retrievePub(pubIDs, numThreads, pub_dict)
    # write objects to output file
    if output is not None:
        with open(output, 'w') as jsonfile:
        	json.dump(results, jsonfile)

    return results

def extractFromList(input, numThreads=1, output=None):
    # initialize variables
    classifiedTools = input
    pubIDs = []
    pub_dict = dict() # keeps track of [pmid, obj(id, title, abstract, is_tool)] values

    # parse values from input file and store in dictionary
    for tool in classifiedTools:
        pubIDs.append(int(tool['pmid']))
        pub_dict[int(tool['pmid'])] = tool

    results = retrievePub(pubIDs, numThreads, pub_dict)
    # write objects to output file
    if output is not None:
        with open(output, 'w') as jsonfile:
        	json.dump(results, jsonfile)

    return results

def retrievePub(pubIDs, numThreads, classifiedPubs={}):
    results = []
    numPubs = len(pubIDs)
    batchSize = int(numPubs/numThreads)
    remainder = numPubs%numThreads
    if batchSize<1:
        numThreads = numPubs
        batchSize = 1
    threads = []
    for i in range(numThreads):
        start_in = int(i*batchSize+remainder)
        if i<=remainder:
            start_in = int(i*(batchSize+1))

        end_in = int(start_in + batchSize)
        if i < remainder:
            end_in = int(start_in + batchSize+1)

        print('Thread', i, start_in,'-',end_in)
        myThread = AbstractDownload(i, results, pubIDs[start_in: end_in], classifiedPubs)
        threads.append(myThread)
        myThread.start()

    for t in threads:
        t.join()

    return results


class AbstractDownload(threading.Thread):
    def __init__(self, threadID, results, pubIDs, classifiedPubs={}):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.results = results
        self.pubIDs = pubIDs
        self.classifiedPubs = classifiedPubs
        print("Thread", self.threadID,"starting")
    def run(self):
        self.results += retrievePub_helper(self.pubIDs, self.classifiedPubs);
        print("Thread", self.threadID,"finished")

def retrievePub_helper(pubIDs, classifiedPubs={}):
    # call the pubmed API with IDs and get the results

    results = [] # return array of objects
    xml_data = [] # data received from query in XML format
    for id in pubIDs:
        tempURL = PUBMED_ID_URL + str(id)
        r = requests.get(tempURL)
        xml_data.append(r.text)
        # print("Retrieving:", id)

    # parse the xml results and just get the article title and abstract
    for data in xml_data:
        root = ET.fromstring(data.encode('utf-8'))
        pmid = root.find('PubmedArticle').find('MedlineCitation').find('PMID').text
        article = root.find('PubmedArticle').find('MedlineCitation').find('Article')
        obj = {}

        if len(classifiedPubs)!=0:
            obj = classifiedPubs[int(pmid)]

        obj['pmid'] = pmid
        obj['title'] = article.find('ArticleTitle').text
        obj['abstract'] = ''
        obj['type'] = []

        abstract_request = requests.get(PUBMED_ABSTRACT_URL.format(str(pmid)))
        obj['abstract'] = ET.fromstring(abstract_request.text).text

        doi = ''

        for id in root.find('PubmedArticle').find('PubmedData').find('ArticleIdList').findall('ArticleId'):
            if id.get('IdType')=='doi':
                doi = id.text
        obj['doi'] = doi

        for type in article.find('PublicationTypeList'):
            obj['type'].append(type.text)

        results.append(obj)
        # print("Retrieved", obj['title'])

    return results
