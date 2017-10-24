# - *- coding: utf- 8 - *-

'''
by M Deutsch
06/05/2017

Script to load data, process it and use it to predict new values.
Contains special functions for the given (private) problem

'''



from __future__ import division
import sys
import argparse
import pickle
import json
import urllib2
import shlex
import cceprocess
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, precision_score, recall_score, log_loss
import pickle
import time


reload(sys)
sys.setdefaultencoding('utf8')


def restricted_float(x):
    '''
    restricted float for argparse, checks if between 0 and 1
    :param x: float
    :return: x
    '''
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

def sort_results(data, sort_by = 'predicted'):
    '''
    sort a list of dictionaries by a certain key given by sort_by

    :param data: list of dictionaries
    :param data: key contained in each dictionary
    '''
    ls = sorted(data, key=lambda k: k[sort_by], reverse=True)
    for l in ls:
        print l['url'].replace(' ','').replace('\n',''), l[sort_by]
    return ls


def report(predicted, true, threshold=0.7):
    '''
    print a report of the prediction model for a given threshold

    :param predicted: numpy array of predicted values
    :param true: numpy array of true values
    :return: F-Score and other measures
    '''

    predicted_class = [0 if pre < threshold else 1 for pre in predicted]
    logloss = log_loss(true, predicted, labels=[0,1])
    f1score = f1_score(true, predicted_class, labels=[0,1])
    f05score = fbeta_score(true, predicted_class, beta=0.5, labels=[0,1])
    f2score = fbeta_score(true, predicted_class, beta=2, labels=[0, 1])
    acc = accuracy_score(true, predicted_class)
    recall = recall_score(true, predicted_class, labels=[0,1])
    precision = precision_score(true, predicted_class, labels=[0, 1])
    false_positives = sum([(p == 1) and (t == 0) for p,t in zip(predicted_class, true)])
    false_negatives = sum([(p == 0) and (t == 1) for p,t in zip(predicted_class, true)])

    print '____________________'
    print 'REPORT OF YOUR MODEL'
    print '--------------------'
    print 'your model predicted', len(predicted), 'values'
    print 'the data consists of', sum(true), 'pearls and', len(true) - sum(true), 'non-pearls'
    print 'the model predicted', sum(predicted_class), 'pearls and', len(predicted_class) - sum(predicted_class), 'non-pearls'
    print 'the threshold was at', threshold
    print 'F0.5-Score: ', f05score, ' (optimal at 1, worst at zero)'
    print 'F2-Score: ', f2score, ' (optimal at 1, worst at zero)'
    print 'Accuracy: ', acc, ' (optimal at 1, worst at zero)'
    print 'Log-Loss: ', logloss, ' (optimal at 0, worst at +oo)'
    print 'F1-Score: ', f1score, ' (optimal at 1, worst at zero)'
    print 'Precision: ', precision, ' (optimal at 1, worst at zero)'
    print 'Recall: ', recall, ' (optimal at 1, worst at zero)'
    print 'Number of False Positives: ', false_positives, ' (out of ', sum(predicted_class), ')'
    print 'Number of False Negatives: ', false_negatives, ' (out of ', len(predicted) - sum(predicted_class), ')'


def format_entities(list_of_strings):
    '''
    formats solr-named entities coming from a list of strings to dictionaries handed to the python webserver

    :param list_of_strings: list of string representing the named entities. corresponds to the format of the rfc4180 column in solr
    :return: list of dicts: list of dicts corresponding to the format given to the python webserver
    '''

    list_of_dicts = []
    if type(list_of_strings) == type([]):
        for string in list_of_strings:
            splitter = shlex.shlex(unicode(string).encode('utf8'), posix = True)
            splitter.whitespace = ','
            splitter.quotes = '"'
            splitter.commenters = ''
            splitter.whitespace_split = True
            if string[0] == 'k':
                entity_dict = dict(zip(['known', 'begin', 'end', 'rtype', 'surfaceform', 'resourceuri', 'label', 'confidence'], list(splitter)))
                entity_dict['confidence'] = float(entity_dict['confidence'])/100
            else:
                entity_dict = dict(zip(['known', 'begin', 'end', 'rtype', 'surfaceform'], list(splitter)))
            entity_dict['begin'] = int(entity_dict['begin'])
            entity_dict['end'] = int(entity_dict['end'])
            del entity_dict['known']
            entity_dict['offset'] = 0
            list_of_dicts = list_of_dicts + [entity_dict]
    return list_of_dicts


def load_solr_data(solrshard, solrquery, rows = 300, save = False, filename = 'data_test'):
    '''

    :param solrshard: url to pull data from
    :param filename: filename to dump data (without .p)
    :param solrquery: solrquery for loading the data, in url-compatible format
    :return: data in dictionary format
    '''

    print '\nGET DATA'
    curl_url = '{0}&rows={1}&wt=json&indent=true'.format(solrshard + solrquery, rows)
    print 'from', curl_url
    response = urllib2.urlopen(curl_url)
    data = json.load(response)
    print '\nLOADED DATA'
    documents = data['response']['docs']
    for doc in documents:
        if 'rfc4180' in doc:
            doc['namedEntities'] = format_entities(doc['rfc4180'])
            for entity_dict in doc['namedEntities']:
                entity_dict['surfaceform'] = unicode(entity_dict['surfaceform'].replace("\r", "").replace("\n", "")).encode('utf-8')
                if 'label' in entity_dict:
                    entity_dict['label'] = unicode(entity_dict['label']).encode('utf-8')
        else:
            doc['namedEntities'] = []
    if save:
        pickle.dump(documents, open(filename+'.p', "wb")) ### later: use the dumped file!
    return documents

def do_predictions(data):
    performance = list()
    i = 0
    predicted_values = []
    true_values = []
    print '----- ' * 10
    print 'GET FEATURES AND PREDICTION'

    for i in xrange(len(data)):
        keys = ['text', 'title', 'url', 'teaser']
        pearl = False
        if 'pearl' in data[i].keys():
            pearl = True
            keys = keys + ['pearl']
        document = {key: unicode(data[i][key]) for key in keys}
        document['namedEntities'] = data[i]['namedEntities']
        document['publicationName'] = data[i]['publicationName']
        document['topics'] = data[i]['topics']
        document['pearl'] = data[i]['pearl']
        document['language'] = 'german' if data[i]['language']=='de' else 'english'
        try:
            document['quotations'] = data[i]['quotations']
        except KeyError:
            document['quotations'] = ''
        document['transform'] = False

        start_time = time.time()
        doc_obj = cceprocess.process(document)
        performance.append(time.time() - start_time)

        perligkeit = doc_obj.output['prediction']['perligkeit']/100
        linguistic_quality = doc_obj.output['prediction']['linguisticQuality']/100
        data[i]['linguistic_quality'] = linguistic_quality
        data[i]['content_quality'] = doc_obj.output['prediction']['contentQuality'] / 100
        data[i]['perligkeit'] = perligkeit
        # prediction is perligkeit
        predicted_values = predicted_values + [perligkeit]
        if pearl:
            true_values = true_values + [int(document['pearl']>0)]
        i += 1
        print i, 'done'
    print '----- ' * 10
    print 'average time per document: {}'.format(float(sum(performance))/len(performance))
    return predicted_values, true_values, data

def sort_results(data, sort_by = 'predicted'):
    ls = sorted(data, key=lambda k: k[sort_by], reverse=True)
    for l in ls:
        print l['url'].replace(' ','').replace('\n',''), l[sort_by]

def report(predicted, true, threshold=0.7):
    '''
    print a report of the prediction model

    :param predicted: numpy array of predicted values
    :param true: numpy array of true values
    :return: F-Score and other measures
    '''

    predicted_class = [0 if pre < threshold else 1 for pre in predicted]
    logloss = log_loss(true, predicted, labels=[0,1])
    f1score = f1_score(true, predicted_class, labels=[0,1])
    f05score = fbeta_score(true, predicted_class, beta=0.5, labels=[0,1])
    f2score = fbeta_score(true, predicted_class, beta=2, labels=[0, 1])
    acc = accuracy_score(true, predicted_class)
    recall = recall_score(true, predicted_class, labels=[0,1])
    precision = precision_score(true, predicted_class, labels=[0, 1])
    false_positives = sum([(p == 1) and (t == 0) for p,t in zip(predicted_class, true)])
    false_negatives = sum([(p == 0) and (t == 1) for p,t in zip(predicted_class, true)])

    print '____________________'
    print 'REPORT OF YOUR MODEL'
    print '--------------------'
    print 'your model predicted', len(predicted), 'values'
    print 'the data consists of', sum(true), 'pearls and', len(true) - sum(true), 'non-pearls'
    print 'the model predicted', sum(predicted_class), 'pearls and', len(predicted_class) - sum(predicted_class), 'non-pearls'
    print 'the threshold was at', threshold
    print 'F0.5-Score: ', f05score, ' (optimal at 1, worst at zero)'
    print 'F2-Score: ', f2score, ' (optimal at 1, worst at zero)'
    print 'Accuracy: ', acc, ' (optimal at 1, worst at zero)'
    print 'Log-Loss: ', logloss, ' (optimal at 0, worst at +oo)'
    print 'F1-Score: ', f1score, ' (optimal at 1, worst at zero)'
    print 'Precision: ', precision, ' (optimal at 1, worst at zero)'
    print 'Recall: ', recall, ' (optimal at 1, worst at zero)'
    print 'Number of False Positives: ', false_positives, ' (out of ', sum(predicted_class), ')'
    print 'Number of False Negatives: ', false_negatives, ' (out of ', len(predicted) - sum(predicted_class), ')'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--threshold', metavar='threshold', type=restricted_float,
        default=0.5, help='the threshold for considering an article a pearl, between 0 and 1')
    threshold = parser.parse_args().threshold

    data = load_solr_data('curator%3A"Test"')
    #pickle.dump( data, open( "curator_data.p", "wb" ) )
    #data = pickle.load(open( "curator_data.p", "rb" ) )
    predicted, true, data = do_predictions(data)
    report(predicted, true, threshold=threshold)
