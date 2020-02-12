import csv
import json
import re
import string
import time

import nltk
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.text import Text

import requests
from bs4 import BeautifulSoup

base_url = 'https://planning.n-somerset.gov.uk/online-applications/applicationDetails.do?activeTab=neighbourComments&keyVal=PJML85LPMKI00&neighbourCommentsPager'


def write_to_csv(result_list, filename):
    writer = csv.DictWriter(open(filename, 'w+', newline=''),
                            fieldnames=['address', 'stance', 'date_submitted', 'text'])
    writer.writeheader()
    writer.writerows(result_list)


def build_sentiment_model(comment_file_name):
    reader = csv.DictReader(open(comment_file_name, 'r'),
                            fieldnames=['address', 'stance', 'date_submitted', 'text'])
    comments = [comment for comment in reader]
    for comment in comments:
        sent_tokens = [re.sub(r'[^\w\s]', '', token) for token in sent_tokenize(comment)]
        comment['sent_tokens_filtered'] = [word for sent in sent_tokens for word in sent if word not in stopwords]
        comment['POS_tag'] = [nltk.pos_tag(tokenized_sent) for tokenized_sent in comment['sent_tokens_filtered']]
        comment['adjectives'] = [tag[0] for tag in comment['POS_tag'] if tag[1] == 'ADJ']

    json.dump(comments, open('word_model.json', 'w+'))


def download_comments(comment_file_name):
    p = 1
    results = []
    while True:
        page = requests.get(base_url + f'.page={p}')
        if page.status_code != 200:
            if results:
                write_to_csv(results, comment_file_name)
            page.raise_for_status()

        soup = BeautifulSoup(page.text, "html.parser")
        comments = soup.find_all("div", class_="comment")
        if not comments:
            break
        for comment in comments:
            address = comment.select_one('h3 span', class_='consultationAddress')
            if address is None:
                break
            results.append({'address': address.text.strip(), 'stance': address.find_next_sibling().text.strip(),
                            'date_submitted': " ".join(comment.select_one('div h4', class_='commentText').text.split()),
                            'text': comment.select_one('div p', class_='commentText').text.strip()})
        p += 1
        time.sleep(2)

    write_to_csv(results, comment_file_name)


if __name__ == '__main__':
    download_comments('comments.csv')
    build_sentiment_model('comments.csv')
