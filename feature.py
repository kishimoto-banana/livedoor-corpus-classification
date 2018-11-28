from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
import gensim
from gensim import corpora, matutils
import numpy as np

class Doc2Vec_feature(TransformerMixin):

    def __init__(self, model):
        self.doc2vec = model

    def fit(self, X, y=None):
        pass
    
    
    def fit_transform(self, X, y=None):

        X_dense = self.transform(X)

        return X_dense
    
    def transform(self, X, y=None):

        doc2vec = self.doc2vec
        
        X_dense = np.zeros((X.shape[0], 100))
        for idx, tokenized_texts in enumerate(X):
             X_dense[idx, :] = doc2vec.infer_vector(tokenized_texts)
        
        return X_dense

class BoW(TransformerMixin):
    '''
    BoWクラス
    '''
    def __init__(self):
        self.dictionary = None

    def fit(self, X, y=None):
        '''
        
        '''
        dictionary = gensim.corpora.Dictionary(X)
        dictionary.filter_extremes(no_below=5, no_above=0.5)
        self.dictionary = dictionary
        
        return self
    
    def fit_transform(self, X, y=None):
        dictionary = gensim.corpora.Dictionary(X)
        dictionary.filter_extremes(no_below=5, no_above=0.5)
        self.dictionary = dictionary
        
        X_dense = self.transform(X)
        
        return X_dense
    
    def transform(self, X, y=None):
        dictionary = self.dictionary
        
        for idx, words in enumerate(X):
            BoW_ = dictionary.doc2bow(words)
            dense = matutils.corpus2dense([BoW_], num_terms=len(dictionary)).T[0]
            if idx == 0:
                X_dense = np.zeros((X.shape[0], len(dense)))
            X_dense[idx,:] = dense
            
        return X_dense

class Wiki2Vec(TransformerMixin):
    '''
    Wikipedia2Vecによるembedding
    '''
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        pass
    
    def fit_transform(self, X, y=None):
        
        X_dense = self.transform(X)
        
        return X_dense
  
    def transform(self, X, y=None):

        wiki2vec = self.model
        X_dense = np.zeros((X.shape[0], 100))
        for idx, words in enumerate(X):         
            entity_count = 0
            X_per_entity = None
            for word in words:
                entity = wiki2vec.get_entity(word.upper())
                if entity is not None:
                    entity_count += 1
                    if X_per_entity is None:
                        X_per_entity = np.array(wiki2vec.get_entity_vector(entity.title))
                    else:
                        X_per_entity += np.array(wiki2vec.get_entity_vector(entity.title))

            if entity_count == 0:
                X_per_entity = np.zeros(100)
            else:
                X_per_entity /= entity_count
                norm = np.linalg.norm(X_per_entity, 2)
                if norm != 0:
                    X_per_entity = X_per_entity / norm

            X_dense[idx, :] = X_per_entity
        
        return X_dense

class BowWiki(TransformerMixin):
    '''
    BoWとwikipedia2vecの連結クラス
    '''
    def __init__(self, model):

        self.bow = BoW()
        self.wiki2vec = Wiki2Vec(model=model)
        self.pca = PCA(n_components=100)

    def fit(self, X, y=None):
        
        self.bow.fit(X)
        return self

    def fit_transform(self, X, y=None):
        
        X_bow = self.bow.fit_transform(X)
        X_bow = self.pca.fit_transform(X_bow)
        X_wiki2vec = self.wiki2vec.fit_transform(X)
        X_dense = np.hstack((X_bow, X_wiki2vec))

        return X_dense

    def transform(self, X, y=None):

        X_bow = self.bow.transform(X)
        X_bow = self.pca.transform(X_bow)
        X_wiki2vec = self.wiki2vec.transform(X)
        X_dense = np.hstack((X_bow, X_wiki2vec))

        return X_dense

class Doc2VecWiki(TransformerMixin):
    '''
    Doc2Vecとwikipedia2vecの連結クラス
    '''
    def __init__(self, doc2vec_model, wiki2vec_model):

        self.doc2vec = Doc2Vec_feature(model=doc2vec_model)
        self.wiki2vec = Wiki2Vec(model=wiki2vec_model)

    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):

        X_dense = self.transform(X)
        return X_dense

    def transform(self, X, y=None):

        X_doc2vec = self.doc2vec.fit_transform(X)
        X_wiki2vec = self.wiki2vec.fit_transform(X)
        X_dense = np.hstack((X_doc2vec, X_wiki2vec))

        return X_dense