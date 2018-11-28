from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.externals import joblib
import numpy as np
from wikipedia2vec import Wikipedia2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import helpers
from feature import Doc2VecWiki
import evaluator

def main():
    # データの読み込み
    df = helpers.load_data()
    df['body_wakati'] = df.body.apply(helpers.fetch_tokenize)

    # 入力データと正解ラベル生成
    X = df.body_wakati.values
    le = LabelEncoder()
    y = le.fit_transform(df.category)

    # doc2vecの学習
    print('training doc2vec')
    training_data = [TaggedDocument(words=tokenize_texts, tags=[idx]) for idx, tokenize_texts in enumerate(X)]
    doc2vec = Doc2Vec(training_data, vector_size=100, workers=4)
    print('finish training doc2vec')

    # Wikipedia2Vecの学習済モデル読み込み
    wiki2vec = Wikipedia2Vec.load('models/jawiki_20180420_100d.pkl')

    # # パイプラインの構築とグリッドサーチ
    pipe = make_pipeline(Doc2VecWiki(doc2vec_model=doc2vec, wiki2vec_model=wiki2vec), SVC(random_state=0, probability=True))
    param_range = [0.1, 1, 10, 100]
    param_grid = [
        {'C':param_range, 'kernel':'linear'},
        {'C':param_range, 'gamma':param_range, 'kernel':'rbf'}
    ]
    best_score, best_model = evaluator.grid_search(estimator=pipe, params=param_grid,X=X, y=y)
    print(best_score)
    print(best_model.get_params())

    # スコアとモデルの保存
    save_dir = './models/doc2vec_wiki'
    helpers.mkdir(save_dir)
    np.savetxt(save_dir + '/accuracy.txt', np.array(best_score).reshape(1,1))    
    joblib.dump(best_model, save_dir + '/model.pkl')

if __name__ == '__main__':
    main()