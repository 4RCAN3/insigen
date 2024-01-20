import pandas as pd
import insigen 
import scraper 
import json

df = pd.read_csv('Data/wiki.csv')
ins = insigen.insigen()

tds = []

for row in df['url'][:3000]:
    article = scraper.getArticle(row)
    td = ins.get_topic_distribution(article)
    tds.append(td)

with open('td.json', 'w') as f:
    json.dump(tds, f)
