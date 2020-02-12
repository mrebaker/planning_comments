import csv
import time

import requests
from bs4 import BeautifulSoup

base_url = 'https://planning.n-somerset.gov.uk/online-applications/applicationDetails.do?activeTab=neighbourComments&keyVal=PJML85LPMKI00&neighbourCommentsPager'


def write_to_csv(result_list):
    writer = csv.DictWriter(open('comments.csv', 'w+', newline=''),
                            fieldnames=['address', 'stance', 'date_submitted', 'text'])
    writer.writeheader()
    writer.writerows(result_list)


if __name__ == '__main__':
    p = 1
    results = []
    while True:
        page = requests.get(base_url + f'.page={p}')
        if page.status_code != 200:
            if results:
                write_to_csv(results)
            page.raise_for_status()

        soup = BeautifulSoup(page.text, "html.parser")
        comments = soup.find_all("div", class_="comment")
        if not comments:
            break
        for comment in comments:
            address = comment.select_one('h3 span', class_='consultationAddress')
            if address is None:
                break
            results.append({'address': address.text.strip(),
                            'stance': address.find_next_sibling().text.strip(),
                            'date_submitted': " ".join(comment.select_one('div h4', class_='commentText').text.split()),
                            'text': comment.select_one('div p', class_='commentText').text.strip()
                            })
        p += 1
        time.sleep(2)

    write_to_csv(results)
