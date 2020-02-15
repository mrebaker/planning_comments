import csv
import json
import random
import re
import time

import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

import pickle
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

base_url = 'https://planning.n-somerset.gov.uk/online-applications/applicationDetails.do?activeTab=neighbourComments&keyVal=PJML85LPMKI00&neighbourCommentsPager'


def basic_analysis():
    df = pd.read_csv('comments.csv', encoding='latin-1', names=['address', 'stance', 'date', 'comment'])
    print(df['stance'].value_counts())


def extract_comments(comments_list):
    return_list = []
    for comment in comments_list:
        address = comment.select_one('h3 span', class_='consultationAddress')
        if address is None:
            return return_list
        return_list.append({'address': address.text.strip(), 'stance': address.find_next_sibling().text.strip(),
                            'date_submitted': " ".join(comment.select_one('div h4', class_='commentText').text.split()),
                            'text': comment.select_one('div p', class_='commentText').text.strip()})
    return return_list


def write_to_csv(result_list, filename):
    writer = csv.DictWriter(open(filename, 'a+', newline=''),
                            fieldnames=['address', 'stance', 'date_submitted', 'text'])
    writer.writerows(result_list)


def build_classifier(data):
    train, test = train_test_split(data, test_size=0.2, shuffle=True)
    classifier = nltk.NaiveBayesClassifier.train(train)
    print(nltk.classify.accuracy(classifier, test) * 100)
    classifier.show_most_informative_features()
    pickle.dump(classifier, open('classifier', 'wb+'))


def find_features(word_list, adj_limit):
    word_features = most_common_adjectives(adj_limit)
    return {w: (w in word_list) for w in word_features}


def build_feature_set(adjective_limit, comment_limit=None):
    comments = json.load(open('word_model.json', 'r'))
    if comment_limit:
        comments = comments[:comment_limit]
    feature_set = [(find_features(comment['adjectives'], adjective_limit), comment['stance'])
                   for comment in comments]
    return np.array(feature_set)


def extract_adjectives(comment_file_name):
    reader = csv.DictReader(open(comment_file_name, 'r'),
                            fieldnames=['address', 'stance', 'date_submitted', 'text'])
    comments = [comment for comment in reader]
    for comment in tqdm(comments):
        sent_tokens = [re.sub(r'[^\w\s]', '', token) for token in sent_tokenize(comment['text'])]
        comment['sent_tokens_filtered'] = [word for sent in sent_tokens
                                           for word in sent.split()
                                           if word not in stopwords.words()]
        comment['POS_tag'] = nltk.pos_tag(comment['sent_tokens_filtered'])
        comment['adjectives'] = [tag[0] for tag in comment['POS_tag'] if tag[1] in ['JJ', 'JJR', 'JJS']]

    json.dump(comments, open('word_model.json', 'w+'))


def most_common_adjectives(n):
    comments = json.load(open('word_model.json', 'r'))
    adjs = [adj.lower() for comment in comments for adj in comment['adjectives']]
    all_words = nltk.FreqDist(adjs)
    return list(all_words.keys())[:n]


def download_comments(comment_file_name):
    session = requests.session()
    session.get(base_url + '.page=1')
    p = 0
    while True:
        p += 1
        page_url = base_url + f'.page={p}'
        page = session.get(page_url)
        if page.status_code != 200:
            page.raise_for_status()
        soup = BeautifulSoup(page.text, "html.parser")
        comments = soup.find_all("div", class_="comment")
        results = extract_comments(comments)
        if not results:
            break
        write_to_csv(results, comment_file_name)
        time.sleep(2)


if __name__ == '__main__':
    # download_comments('comments.csv')
    # n_adjectives = 1000
    # build_classifier(build_feature_set(n_adjectives))
    basic_analysis()