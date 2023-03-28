import streamlit as st
import pandas as pd
import spacy as sp
import bs4 as bs4
import spacy
import requests
import string
import json
import lxml
import nltk
import re
import os
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer   
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from decouple import config
from PIL import Image

sp.prefer_gpu()

class ScrapTool:
    def visit_url(self, website_url):
        headers = { 
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600',
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
        }
        content = requests.get(website_url,headers).content
        soup = BeautifulSoup(content, "lxml")
        result = {
            "website_url": website_url,
            "website_name": self.get_website_name(website_url),
            "website_description": self.get_html_title_tag(soup)+self.get_html_meta_tags(soup)+self.get_html_heading_tags(soup)+
                                                               self.get_text_content(soup)
        }
        return pd.Series(result)
    
    def content_url(self, website_url):
        headers = { 
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600',
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
        }
        content = requests.get(website_url,headers).content
        soup = BeautifulSoup(content, "lxml")
        result = {
            "website_name": self.get_website_name(website_url),
            "website_title": self.get_html_title_tag(soup),
            "website_description": self.get_html_meta_tags(soup)
        }
        return pd.Series(result)
    
    def image_url(self, website_url):
        headers = { 
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600',
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
        }
        content = requests.get(website_url,headers).content
        soup = BeautifulSoup(content, "lxml")
        result = {
            "website_image": self.get_website_image(soup)
        }
        return pd.Series(result)
    
    def get_website_name(self,website_url):
        return "".join(urlparse(website_url).netloc.split(".")[-2])
    
    def get_html_title_tag(self,soup):
        return '. '.join(soup.title.contents)
    
    def get_html_meta_tags(self,soup):
        tags = soup.find_all(lambda tag: (tag.name=="meta") & (tag.has_attr('name') & (tag.has_attr('content'))))
        content = [str(tag["content"]) for tag in tags if tag["name"] in ['keywords','description']]
        return ' '.join(content)
    
    def get_html_heading_tags(self,soup):
        tags = soup.find_all(["h1","h2","h3","h4","h5","h6"])
        content = [" ".join(tag.stripped_strings) for tag in tags]
        return ' '.join(content)
    
    def get_website_image(self,soup):
        if soup.find("meta", property="image"):
            image = soup.find("meta", property="image").get('content')
        elif soup.find("meta", property="og:image"):
            image = soup.find("meta", property="og:image").get('content')
        elif soup.find("img", src=True):
            image = soup.find("img").get('src')
        return ''.join(image)
    
    def get_text_content(self,soup):
        tags_to_ignore = ['style', 'script', 'head', 'title', 'meta', '[document]',"h1","h2","h3","h4","h5","h6","noscript"]
        tags = soup.find_all(string=True)
        result = []
        for tag in tags:
            stripped_tag = tag.strip()
            if tag.parent.name not in tags_to_ignore\
                and isinstance(tag, bs4.element.Comment)==False\
                and not stripped_tag.isnumeric()\
                and len(stripped_tag)>0:
                result.append(stripped_tag)
        return ' '.join(result)
    
def call_image(url_scraping):
    # call the api
    scrapTool = ScrapTool()
    web_image=scrapTool.image_url(url_scraping)
    df_image= pd.DataFrame(web_image)
    print('df_image:',df_image)
    result_image = df_image.to_json(orient="records") 
    image_url = web_image['website_image']
    return image_url

def call_description(url_scraping):
    # call the api
    scrapTool = ScrapTool()
    web_title=dict(scrapTool.content_url(url_scraping))
    return web_title

if "API_ENDPOINT" not in os.environ:
    os.environ["API_ENDPOINT"] = config("API_ENDPOINT")
    
def call_api(url_scraping):
    # call the api
    scrapTool = ScrapTool()
    web_content=dict(scrapTool.visit_url(url_scraping))
    web_content1=scrapTool.visit_url(url_scraping)
    df_web= pd.DataFrame(web_content1)
    df1_t = df_web.T 
    result1 = df1_t.to_json(orient="records")  
    
    headers1 = {
        'Content-Type': 'application/json'
    }
    url = os.environ["API_ENDPOINT"] + "/url/predict"
    response = requests.request("POST", url, headers=headers1, data=result1)
    results = response.json()
    return results

st.set_page_config(
    page_title="LinkScribe - Website scraping and classifier",
    layout="wide"
)

st.title("LinkScribe")

st.write("This is a project by web scraping and classifying the content: Fullstack AI projects")

url_scraping = st.text_input("Insert a URL", value = "https://www.disneyplus.com/")

i_was_clicked = st.button("Inference Classifier")
if i_was_clicked:
    # call the api
    results = call_api(url_scraping)
    st.write(results)

i_was_clicked_title = st.button("Description Website")
if i_was_clicked_title:
    # call the description
    results = call_description(url_scraping)
    st.write(results)

i_was_clicked_image = st.button("Image Website")
if i_was_clicked_image:
    # call the image
    results = call_image(url_scraping)
    st.image(
        results,
        width=600, 
        caption='Thumbnail Image'
        )
    
#st.write("¡Save your web pages by logging in!")

#with st.form("my_form"):
#    username = st.text_input("Username")
#    password = st.text_input("Password")
#    st.form_submit_button("Login")

st.caption('Website by Samantha Gallego, Jessica Arias & Sebastian Anderson')
