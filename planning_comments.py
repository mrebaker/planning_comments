from bs4 import BeautifulSoup
import csv
import requests

base_url = 'https://planning.n-somerset.gov.uk/online-applications/applicationDetails.do?activeTab=neighbourComments&keyVal=PJML85LPMKI00&neighbourCommentsPager'

if __name__ == '__main__':
    p = 1
    results = []
    while True:
        page = requests.get(base_url + f'.page={p}')
        if page.status_code != 200:
            page.raise_for_status()
        soup = BeautifulSoup(page.text, "html.parser")
        comments = soup.find_all("div", class_="comment")
        if comments:
            for comment in comments:
                address = comment.select_one('h3 span', class_='consultationAddress')
                results.append({'address': address.text.strip(),
                                'stance': address.find_next_sibling().text.strip(),
                                'date_submitted': " ".join(comment.select_one('div h4', class_='commentText').text.split()),
                                'text': comment.select_one('div p', class_='commentText').text.strip()
                                })
            writer = csv.DictWriter(open('comments.csv', 'w+'), ['address', 'stance', 'date_submitted', 'text'])
            writer.writerows(results)
            exit(0)
            p += 1
        else:
            break
