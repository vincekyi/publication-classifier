import json
import utilities.extract as ex
import utilities.mongo as mongo
from time import sleep
import numpy as np
import csv
import requests


def update(mongoUrl, data_endpoint):
    # connect to mongo db
    db  = mongo.DBClient(mongoUrl)

    # #############################################################
    # Extract abstracts from Pubmed and insert into database
    # ############################################################


    # request classification data from endpoint
    req = requests.get(data_endpoint)
    obj = json.loads(req.text)
    print('Number of ambiguous:', len(obj['ambiguous']))
    print('Number of tools:',len(obj['tools']))
    print('Number of nontools', len(obj['not tools']))

    # prepare each training example
    classifiedTools = []
    if obj is not None:
        if obj['tools'] is not None:
            for tool in obj['tools']:
                new_tool = {}
                new_tool['pmid'] = tool['pmid']
                new_tool['journal'] = tool['journal']
                new_tool['fullTextViewed'] = True if tool['fulltextviewed']==1 else False
                new_tool['isTool'] = True
                classifiedTools.append(new_tool)

        if obj['not tools'] is not None:
            for tool in obj['not tools']:
                new_tool = {}
                new_tool['pmid'] = tool['pmid']
                new_tool['journal'] = tool['journal']
                new_tool['fullTextViewed'] = True if tool['fulltextviewed']==1 else False
                new_tool['isTool'] = False
                classifiedTools.append(new_tool)


    publications = ex.extractFromList(classifiedTools, 12)
    print('Number of publications to be updated:', len(publications))
    for pub in publications:
        result = db.insertTool(pub)
        if not result.acknowledged:
            print('Could not insert', pub['pmid'])
    print('Finished update')

    # post update analysis
    before = []
    for tool in classifiedTools:
        before.append(tool['pmid'])
    after = []
    for tool in publications:
        after.append(tool['pmid'])

    setBefore = set(before)
    setAfter = set(after)
    print('Did not insert:')
    print(list(setBefore.difference(setAfter)))


# #############################################################
if __name__ == '__main__':
    update('mongodb://10.44.115.120:27017/pub', 'http://10.44.115.120:8000/classify/data/')
