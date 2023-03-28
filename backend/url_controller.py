import typing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.model_selection import train_test_split  
import numpy as np
import pandas as pd
from fastapi import Depends
from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter
from starlette.requests import Request
from pydantic import BaseModel
from model_loader import ModelLoader
from bs4 import BeautifulSoup
import lxml
import string
import requests
import json
import nltk
import re

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet') 

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

dataset=pd.read_csv('website_classification.csv')

router = InferringRouter()

class UrlModelEntry(BaseModel):
    urlscraping: object

    def to_list(self):
        return [
            self.urlscraping
        ]

async def get_model(req: Request):
    return req.app.state.model

@cbv(router)
class UrlController:
    model: ModelLoader = Depends(get_model)

    @router.get("/")
    def welcome(self):
        return {"message": "Welcome to the URL scraping and classifier API"}

    @router.get("/info")
    def model_info(self):
        """Return model information, version, how to call"""
        return {"name": self.model.name, "version": self.model.version, "carpeta": self.model.model_dir}

    @router.get("/health")
    def service_health(self):
        """Return service health"""
        return "ok"

    @router.post("/predict")
    async def predict(self, request: Request):
        df = dataset[['website_url','cleaned_website_text','Category']].copy()
        df['category_id'] = df['Category'].factorize()[0]
        category_id_df = df[['Category', 'category_id']].drop_duplicates()
        category_to_id = dict(category_id_df.values)
        id_to_category = dict(category_id_df[['category_id', 'Category']].values)
        X = df['cleaned_website_text'] 
        y = df['Category'] 
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=0.25,
                                                            random_state = 42)
        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
                                ngram_range=(1, 2), 
                                stop_words='english')
        fitted_vectorizer = tfidf.fit(X_train)
        request_body = await request.json()
        df_a = pd.DataFrame(request_body)
        df_a['website_description']= df_a['website_description'].apply(lambda x:x.lower())
        df_a['tokenized_words'] = df_a['website_description'].apply(lambda x:word_tokenize(x))
        df_a['tokenized_words'] = df_a['tokenized_words'].apply(lambda x:[re.sub(f'[{string.punctuation}]+','',i) for i in x if i not in list(string.punctuation)])
        df_a['tokenized_words'] = df_a['tokenized_words'].apply(lambda x:' '.join(x))
        X = df_a['tokenized_words']
        t= fitted_vectorizer.transform(X)
        predictions = self.model(t)
        category_int = int(predictions)
        category_predict = id_to_category[category_int]
        return {"predictions": category_predict}
