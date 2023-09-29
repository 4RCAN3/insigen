from bs4 import BeautifulSoup
import requests
import json
import pandas as pd
import random

wiki = 'https://en.wikipedia.org/'

def fetchTopics():
    wikiContents = requests.get('https://en.wikipedia.org/wiki/Wikipedia:Contents')

    soup = BeautifulSoup(wikiContents.content, 'html.parser')

    paragraphs = soup.findAll('p')
    topicParagraphs = paragraphs[4:8] + paragraphs[10:15]
    links = []
    for topicPara in topicParagraphs:
        link = topicPara.find_all('a', href = True)[0]['href']
        links.append(link)

    return links

def getSubtopics():
    topics = fetchTopics()
    subTopics = {}

    for topic in topics:
        topicLink = wiki + topic
        topicContent = requests.get(topicLink)
        soup = BeautifulSoup(topicContent.content, 'html.parser')
        overview = soup.find("div", {"class": "contentsPage__section"})
        overview = str(overview).split('\n')
        
        for subTopic in overview:
            if '–' not in subTopic:
                topicLinks = BeautifulSoup(subTopic, 'html.parser')
            else:
                topicTitle = BeautifulSoup(subTopic.split('–')[0], 'html.parser')
                topicTitle = topicTitle.text.replace('\xa0', '')
                subTopics[topicTitle] = []
                topicLinks = BeautifulSoup(subTopic.split('–')[1], 'html.parser')

            for link in topicLinks.find_all('a', href = True):
                if "Portal" not in link['href']:
                    subTopics[topicTitle].append(link['href'])
        
    
    return subTopics
                  
def writeLinks():
    topics = getSubtopics()
    with open ('urls.txt', 'w') as f:
        topicWriter = json.dumps(topics, indent=4)
        f.write(topicWriter)

def getArticle(page):
    url = wiki + page
    wikiPage = requests.get(url)
    soup = BeautifulSoup(wikiPage.content, 'html.parser')

    paras = soup.find_all('p')
    para = " ".join([para.text for para in paras])

    return para

def CreateDataset():
    urls = open('urls.txt').read()
    urls = json.loads(urls)
    data = [["text", "category_label", "url"]]
    for topic in urls.keys():
        print(topic)
        for page in urls[topic]:
            text = getArticle(page)
            data.append([text, topic, page])

    df = pd.DataFrame(data[1:], columns=data[0])
    df.to_csv('wiki.csv')


def getMore(page, existingLinks):
    url = wiki + page
    wikiPage = requests.get(url)
    soup = BeautifulSoup(wikiPage.content, 'html.parser')
    all_links = soup.find_all('a', href=True)

    prefix = '/wiki/'
    specific_links = [link['href'] for link in all_links if link['href'].startswith(prefix) and ":" not in link['href'] and link['href'] != '/wiki/Main_Page' and link['href'] not in existingLinks]

    return specific_links


def CreateUrls():
    urls = open('urls.txt').read()
    urls = json.loads(urls)
    threshold = 80
    data = {key: [] for key in urls.keys()}
    allLinks = []
    for topic in urls.keys():
        data[topic].extend(urls[topic])
        allLinks.extend(urls[topic])
        if len(urls[topic]) >= threshold:
            continue

        moreUrls = []
        for link in urls[topic]:
            moreLinks = getMore(link, allLinks)
            moreUrls.extend(moreLinks)
            allLinks.extend(moreLinks)

        random.shuffle(moreUrls)
        remaining = threshold - len(urls[topic])
        data[topic].extend(moreUrls[:remaining])

    

    with open ('urls.txt', 'w') as f:
        topicWriter = json.dumps(data, indent=4)
        f.write(topicWriter)
