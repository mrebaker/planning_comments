import csv
from datetime import datetime as dt
import json
import re
import sqlite3
import time

import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, plot_confusion_matrix
import pickle
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
from tqdm import tqdm

base_url = 'https://planning.n-somerset.gov.uk/online-applications/applicationDetails.do?activeTab=neighbourComments&keyVal=PJML85LPMKI00&neighbourCommentsPager'


def basic_analysis():
    df = pd.read_csv('comments.csv', encoding='latin-1', names=['address', 'stance', 'date', 'comment'])
    print(df['stance'].value_counts())


def comments_to_db(comment_file_name):
    reader = csv.DictReader(open(comment_file_name, 'r'), fieldnames=['address', 'stance', 'date_submitted', 'text'])
    conn = sqlite3.connect('comments.db')
    conn.execute('''DROP TABLE IF EXISTS comments''')
    conn.execute('''CREATE TABLE comments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        address TEXT,
                        stance INTEGER,
                        comment_text TEXT,
                        date_submitted TEXT)''')
    conn.commit()
    for comment in reader:
        date_str = dt.strptime(comment['date_submitted'], 'Comment submitted date: %a %d %b %Y').strftime('%Y-%m-%d')
        conn.execute('INSERT INTO comments (address, stance, comment_text, date_submitted)'
                     'VALUES (?, ?, ?, ?)',
                     (comment['address'], comment['stance'], comment['text'], date_str))

    conn.commit()
    db_cleanup()


def db_cleanup():
    conn = sqlite3.connect('comments.db')
    conn.execute('''UPDATE comments SET stance = CASE stance 
                    WHEN '(Objects)' THEN -1 
                    WHEN '(Supports)' THEN 1 
                    WHEN '(Neutral)' THEN 0 END''')
    conn.commit()


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


def locale_analysis(locale):
    conn = sqlite3.connect('comments.db')
    table = conn.execute("""SELECT count(*), stance, CASE WHEN instr(lower(address), lower(:loc)) THEN :loc 
                            ELSE 'Outside ' || :loc  END as locale
                            FROM comments
                            GROUP BY stance, locale""", {'loc': locale}).fetchall()
    # heatmap
    df = pd.DataFrame(table, columns=['comments', 'stance', 'locale'])
    df = df.pivot(index='locale', columns='stance')
    df.columns.set_levels(['oppose', 'neutral', 'support'], level=1, inplace=True)
    print(df)
    fig = go.Figure(go.Heatmap(z=df.values, x=df.columns.get_level_values(1), y=df.index, colorscale='teal'))
    fig.show()

    # stacked bar
    df2 = pd.DataFrame(table, columns=['comments', 'stance', 'locale'])
    fig2 = px.bar(df2, x='locale', y='comments', color='stance')
    fig2.show()

    # 100% stacked bar
    table2 = conn.execute("""SELECT A.comment_count, A.stance, A.locale, 
                                    (1.0 * A.comment_count / B.comment_total) as comment_ratio
                              FROM (SELECT count(*) as comment_count, stance, 
                                        CASE WHEN instr(lower(address), lower(:loc)) THEN :loc 
                                        ELSE 'Outside ' || :loc END as locale
                                    FROM comments 
                                    GROUP BY stance, locale) AS A
                              LEFT JOIN (SELECT count(*) as comment_total, stance, 
                                            CASE WHEN instr(lower(address), lower(:loc)) THEN :loc 
                                            ELSE 'Outside ' || :loc END as locale
                                         FROM comments 
                                         GROUP BY locale) AS B
                              ON A.locale = B.locale""", {'loc': locale}).fetchall()
    df3 = pd.DataFrame(table2, columns=['comment_count', 'stance', 'locale', 'comment_ratio'])
    fig3 = px.bar(df3, x='locale', y='comment_ratio', color='stance')
    fig3.show()


def write_to_csv(result_list, filename):
    writer = csv.DictWriter(open(filename, 'a+', newline=''),
                            fieldnames=['address', 'stance', 'date_submitted', 'text'])
    writer.writerows(result_list)


def build_classifier(n_adj, n_comm, classifier):
    try:
        data = np.load(f'data/data_{n_adj}_{n_comm}', allow_pickle=True)
    except FileNotFoundError:
        data = build_feature_set(n_adj, n_comm)
        data.dump(f'data/data_{n_adj}_{n_comm}')
    train, test = train_test_split(data, test_size=0.2, shuffle=True)
    trained_model = classifier.train(train)
    pickle.dump(trained_model, open(f'classifier-{classifier.__name__}', 'wb+'))
    evaluate_classifier(trained_model, test)
    return trained_model


def evaluate_classifier(classifier, test_data):
    X_test = test_data[:, 0]
    y_true = test_data[:, 1]
    y_pred = classifier.classify_many(test_data[:, 0])
    print('accuracy: ', nltk.classify.accuracy(classifier, test_data) * 100)
    if classifier.__class__ == 'nltk.classify.naivebayes.NaiveBayesClassifier':
        classifier.show_most_informative_features()
    else:
        print(classifier.__class__)
    print('f1 score: ', f1_score(y_true, y_pred, labels=['object', 'support', 'neutral'], average='weighted'))
    print(confusion_matrix(y_true, y_pred, normalize='true'))
    try:
        disp = plot_confusion_matrix(classifier, X_test, y_true,
                                     display_labels=['Neutral', 'Object', 'Support'],
                                     cmap=plt.cm.Blues,
                                     normalize='true'
                                     )
    except ValueError:
        print('Classifier is not from sklearn - cannot use inbuilt plotting function')
        # todo - make this work with nltk classifiers
        return
    plt.show()


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
    comments_to_db(comment_file_name)


if __name__ == '__main__':
    # download_comments('comments.csv')
    locale_analysis('Bristol')
