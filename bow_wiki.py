from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.externals import joblib
import numpy as np
from wikipedia2vec import Wikipedia2Vec
import helpers
from feature import BowWiki
import evaluator

def main():
    # データの読み込み
    df = helpers.load_data()
    df['body_wakati'] = df.body.apply(helpers.fetch_tokenize)

    # 入力データと正解ラベル生成
    X = df.body_wakati.values
    le = LabelEncoder()
    y = le.fit_transform(df.category)

    # Wikipedia2Vecの学習済モデル読み込み
    model = Wikipedia2Vec.load('models/jawiki_20180420_100d.pkl')

    # パイプラインの構築とグリッドサーチ
    pipe = make_pipeline(BowWiki(model=model), SVC(random_state=0, probability=True))
    param_range = [0.1, 1, 10, 100]
    param_grid = [
        {'C':param_range, 'kernel':'linear'},
        {'C':param_range, 'gamma':param_range, 'kernel':'rbf'}
    ]
    best_score, best_model = evaluator.grid_search(estimator=pipe, params=param_grid,X=X, y=y)
    print(best_score)
    print(best_model.get_params())

    # スコアとモデルの保存
    save_dir = './models/bow_wiki'
    helpers.mkdir(save_dir)
    np.savetxt(save_dir + '/accuracy.txt', np.array(best_score).reshape(1,1))    
    joblib.dump(best_model, save_dir + '/model.pkl')

if __name__ == '__main__':
    main()