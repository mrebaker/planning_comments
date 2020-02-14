import csv
import json
import re
import string
import time

import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.text import Text

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

base_url = 'https://planning.n-somerset.gov.uk/online-applications/applicationDetails.do?activeTab=neighbourComments&keyVal=PJML85LPMKI00&neighbourCommentsPager'


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
    build_sentiment_model('comments.csv')
