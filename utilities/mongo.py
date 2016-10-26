from pymongo import MongoClient
import string


class DBClient:
    def __init__(self, mongo_url):
        self.client = MongoClient(mongo_url)
        db_name = mongo_url[(mongo_url.rindex('/')+1):]
        self.db = self.client[db_name]

    def queryDOI(self, doi):
        cursor = self.db.training_set.find({'doi':doi})
        results = []
        for doc in cursor:
            results.append(doc)
        return results

    def queryDOIList(self, dois):
        cursor = self.db.training_set.find({'doi': {'$in': dois}})
        results = []
        for doc in cursor:
            results.append(doc)
        return results

    def queryPMID(self, pmid):
        cursor = self.db.training_set.find({'pmid':pmid})
        results = []
        for doc in cursor:
            results.append(doc)
        return results

    def queryPMIDList(self, pmids):
        cursor = self.db.training_set.find({'pmid': {'$in': pmids}})
        results = []
        for doc in cursor:
            results.append(doc)
        return results

    def queryAll(self):
        cursor = self.db.training_set.find({})
        results = []
        for doc in cursor:
            results.append(doc)
        return results

    def insertTool(self, tool):
        result = self.db.training_set.update_one({'pmid':tool['pmid']},
            {'$set':tool},
            upsert=True)
        return result

    def insertStat(self, stat):
        result = self.db.model.insert_one(stat);
        return result

    def queryStats(self):
        cursor = self.db.model.find({})
        results = []
        for doc in cursor:
            results.append(doc)
        return results
