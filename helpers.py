from pathlib import Path
import pandas as pd
import requests
import MeCab
import os

def load_data():
    '''
    データをロード
    '''
    path = Path('data/')
    categories = [str(candidate).split('/')[-1] for candidate in path.glob('*') if candidate.is_dir()]
    
    docs = []
    for category in categories:
        path = Path(f'data/{category}/')
        filenames = path.glob('*.txt')
        for filename in filenames:
            if str(filename).split('/')[-1] == 'LICENSE.txt':
                continue
                
            texts = filename.read_text()
            texts_split = texts.split('\n')
            url = texts_split[0]
            date = texts_split[1]
            title = texts_split[2]
            body = ''.join(texts_split[3:]).strip()
            
            docs.append((category, url, date, title, body))

    df = pd.DataFrame(docs, columns=['category', 'url', 'date', 'title', 'body'])
    print(df.category.value_counts())
    return df


url = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
response = requests.get(url)
stopwords = [w for w in response.content.decode().split('\r\n') if w != '']

tagger = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/')
target_parts_of_speech = ('名詞')

def fetch_tokenize(text):
    '''
    分かち書き
    '''
    tokenized_text = []
  
    for chunk in tagger.parse(text).splitlines()[:-1]:
        try:
            (surface, feature) = chunk.split('\t')
        except:
            continue
        
        if feature.startswith(target_parts_of_speech):
            if surface not in stopwords:
                tokenized_text.append(surface.lower())


    return tokenized_text

def mkdir(target_dir):
    '''
    ディレクトリ作成
    '''
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)