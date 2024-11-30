#!/usr/bin/env python
# coding: utf-8

# import os
# from pyspark.ml.feature import HashingTF, IDF
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import explode, split, collect_list, concat_ws, lit,trim, col,udf
# from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType, DoubleType
# from pyspark.ml.linalg import SparseVector, VectorUDT, DenseVector
# from pyspark.sql import functions as F
# from pyspark.sql.window import Window
# from sklearn.feature_extraction.text import TfidfVectorizer
# from pyspark.sql.types import StringType
# import numpy as np
# import json
# from sklearn.metrics.pairwise import cosine_similarity
# import pandas as pd
# from itertools import combinations
# from tqdm import tqdm
# from graphframes import GraphFrame

# In[2]:


# !which python
# !pip list


# In[3]:


class Dataset:
    def __init__(self, spark_session, limit = None):
        self.spark_session = spark_session
        self.limit = limit

        # Data files
        self.stackoverflow_posts_df = None
        self.tags_df = None
        self.posts_tag_wiki_excerpt_df = None
        self.posts_tag_wiki_df = None

        self.read_big_query_tables()
        self.clean_dfs()
        return

    # Read all the data sources
    def read_big_query_tables(self):
        self.stackoverflow_posts_df = self.spark_session.read.format("bigquery").option("table", "bigquery-public-data.stackoverflow.stackoverflow_posts").load()
        self.tags_df = self.spark_session.read.format("bigquery").option("table", "bigquery-public-data.stackoverflow.tags").load()
        self.posts_tag_wiki_excerpt_df = self.spark_session.read.format("bigquery").option("table", "bigquery-public-data.stackoverflow.posts_tag_wiki_excerpt").load()
        self.posts_tag_wiki_df = self.spark_session.read.format("bigquery").option("table", "bigquery-public-data.stackoverflow.posts_tag_wiki").load()
        
        if self.limit != None:
            self.stackoverflow_posts_df = self.stackoverflow_posts_df.select("id", "title", "body", "tags", "parent_id", "score").limit(self.limit)
            self.tags_df = self.tags_df.limit(self.limit)
            self.posts_tag_wiki_excerpt_df = self.posts_tag_wiki_excerpt_df.select("id", "body").limit(self.limit)
            self.posts_tag_wiki_df = self.posts_tag_wiki_df.select("id", "body").limit(self.limit)
        else:
            self.stackoverflow_posts_df = self.stackoverflow_posts_df.select("id", "title", "body", "tags", "parent_id", "score")
            self.posts_tag_wiki_excerpt_df = self.posts_tag_wiki_excerpt_df.select("id", "body")
            self.posts_tag_wiki_df = self.posts_tag_wiki_df.select("id", "body")
        
        return
    
    def clean_dfs(self):
        self.stackoverflow_posts_df = self.stackoverflow_posts_df.filter(self.stackoverflow_posts_df.tags != '')
        
        return

    # Returns a dataframe that serves as a lookup table for tag_id -> post_id
    def build_tag_post_df(self):

        tags_posts_df = self.stackoverflow_posts_df.select(self.stackoverflow_posts_df.id.alias("post_id"), explode(split(self.stackoverflow_posts_df.tags, "\\|")).alias("tag_name")).drop("tags")
        tags_posts_df = tags_posts_df.groupBy("tag_name").agg(collect_list("post_id").alias("post_ids"))
        tags_posts_df = tags_posts_df.withColumn("post_ids", concat_ws(",", tags_posts_df.post_ids))
        
        tags_df = self.tags_df.select(self.tags_df.id.alias("tag_id"), "tag_name", "count", "excerpt_post_id", "wiki_post_id")
        tag_post_df = tags_df.join(tags_posts_df, on = 'tag_name', how = 'inner')
        tag_post_df = tag_post_df.limit(10)
        return tag_post_df
        
    # Returns a dataframe that serves as a lookup table for post_id -> post_title & post_body
    def build_post_text_df(self):
        
        posts_texts_df = self.stackoverflow_posts_df.select(self.stackoverflow_posts_df.id.alias("post_id"), "body", "title", "score")
        
        excerpts_texts_df = self.posts_tag_wiki_excerpt_df.select(self.posts_tag_wiki_excerpt_df.id.alias("post_id"), "body")
        excerpts_texts_df = excerpts_texts_df.withColumn("title", lit(None))
        excerpts_texts_df = excerpts_texts_df.withColumn("score", lit(None))

        wikis_texts_df = self.posts_tag_wiki_df.select(self.posts_tag_wiki_df.id.alias("post_id"), "body")
        wikis_texts_df = wikis_texts_df.withColumn("title", lit(None))
        wikis_texts_df = wikis_texts_df.withColumn("score", lit(None))
        
        post_text_df = posts_texts_df.union(excerpts_texts_df).union(wikis_texts_df)
        post_text_df = post_text_df.limit(10)
        return post_text_df


# In[4]:


try:
    spark_session.stop()
except NameError:
    print("Spark session not yet defined.")
#spark_session = SparkSession.builder.appName("532 Project").getOrCreate()
spark_session = SparkSession.builder \
    .appName("532 Project") \
    .config("spark.jars", "gs://532-dataset/notebooks/jupyter/532/graphframes-0.8.2-spark3.0-s_2.12.jar") \
    .config("spark.hadoop.fs.defaultFS", "gs://532-dataset") \
    .getOrCreate()


# In[5]:


# Set checkpoint directory once
spark_session.sparkContext.setCheckpointDir("gs://532-dataset/notebooks/jupyter/532/spark-checkpoints")
# Print configurations to verify
jars = spark_session.sparkContext.getConf().get("spark.jars")
print("JARs added to the environment:", jars)
fs_default = spark_session.sparkContext.getConf().get("spark.hadoop.fs.defaultFS")
print("Default Filesystem:", fs_default)
dataset = Dataset(spark_session)

# Build required dataframes
tag_post_df = dataset.build_tag_post_df()
post_text_df_raw = dataset.build_post_text_df()


# In[ ]:


import nltk
import os

# Define a custom path for nltk_data
nltk_data_dir = os.path.expanduser("/home/nltk_data")
# nltk_data_dir = os.path.expanduser("gs://" + BUCKET_NAME + "/nltk_data")
print(nltk_data_dir)
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Set NLTK data path
nltk.data.path.append(nltk_data_dir)

nltk.download('wordnet', download_dir = nltk_data_dir, quiet = True)
nltk.download('stopwords', download_dir = nltk_data_dir, quiet = True)
nltk.download('punkt', download_dir = nltk_data_dir, quiet = True)
nltk.download('punkt_tab', download_dir = nltk_data_dir, quiet = True)


# In[ ]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType

import re
import string
from bs4 import BeautifulSoup

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))


# In[ ]:


# Define the UDF function outside the class
def clean_text(text):
    text = TextPreprocessor.remove_html_tags(text)
#     text = TextPreprocessor.tokenize_text(text)
    text = TextPreprocessor.normalize_text(text)
    text = TextPreprocessor.remove_urls(text)
    text = TextPreprocessor.remove_stopwords(text)
    text = TextPreprocessor.stem_text(text)
    text = TextPreprocessor.lemmatize_text(text)
    return text
def build_dataset(spark_session):
    # Build the required dataframes (running locally)
    # Note: running with is_local as True without google cloud SDK setup might throw errors
    dataset = Dataset(spark_session)

    tag_post_df = dataset.build_tag_post_df()
    post_text_df_raw = dataset.build_post_text_df()

    print('Schema of tag_post_df:')
    # tag_post_df.printSchema()

    print('-' * 50)

    print('Schema of post_text_df_raw:')
    # post_text_df_raw.printSchema()

    print('Number of tags in the raw dataset: ', tag_post_df.count())
    print('Number of posts in the raw dataset: ', post_text_df_raw.count())

    return tag_post_df, post_text_df_raw
# Create UDF once
clean_text_udf = udf(clean_text, StringType())

class TextPreprocessor:
    def __init__(self, post_text_df_raw):
        self.post_text_df_raw = post_text_df_raw
        return

    # Remove HTML tags
    @staticmethod
    def remove_html_tags(text):
        filtered_html_text = ""
        try:
            filtered_html_text = BeautifulSoup(text, "html.parser").get_text()
        except:
            filtered_html_text = ""
        return filtered_html_text

    @staticmethod
    def normalize_text(text):
        # Remove leading/trailing whitespaces
        text = text.strip()
        
        # Lower the text
        text = text.lower() 
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove non-ASCII characters
        text = re.sub(r'[^\x00-\x7f]', ' ', text)
        
        return text
    
    # Remove URLs
    @staticmethod
    def remove_urls(text):
        return re.sub(r'http\S+', '', text)

    # Remove stopwords
    @staticmethod
    def remove_stopwords(text):
        return ' '.join([word for word in text.split() if word not in stop_words])

    # Tokenize text
    @staticmethod
    def tokenize_text(text):
        return ' '.join(word_tokenize(text))

    # Stemming
    @staticmethod
    def stem_text(text):
        stemmer = PorterStemmer()
        return ' '.join([stemmer.stem(word) for word in text.split()])

    # Lemmatization
    @staticmethod
    def lemmatize_text(text):
        lemmatizer = WordNetLemmatizer()
        return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    def preprocess_text(self):
        # Preprocess title & body of the posts
#         self.post_text_df_raw.show(10)
        self.post_text_df_raw = self.post_text_df_raw.na.drop(subset = ["title", "body"])
        post_text_df_pre_processed = self.post_text_df_raw.withColumn("body", clean_text_udf(col('body')))
        post_text_df_pre_processed = post_text_df_pre_processed.withColumn("title", clean_text_udf(col('title')))
        return post_text_df_pre_processed


# In[ ]:


textPreprocessor = TextPreprocessor(post_text_df_raw)
post_text_df = textPreprocessor.preprocess_text()


# In[ ]:


print('Schema of post_text_df:')
post_text_df.printSchema()
post_text_df = post_text_df.filter(post_text_df["score"] >= 1)
post_text_df = post_text_df.filter(
    (trim(col("title")).isNotNull()) & (trim(col("title")) != "") & 
    (trim(col("body")).isNotNull()) & (trim(col("body")) != "")
)
post_text_df = post_text_df.limit(10)
post_text_df.show()


# In[ ]:


def run_community_detection(vertices,edges):
    graph = GraphFrame(vertices, edges)
    result = graph.connectedComponents()
    # result.show()
    print('Number of communities',result.select("component").distinct().count())
    result = graph.labelPropagation(maxIter=5)
    # result.show()   
    print('Number of communities', result.select("label").distinct().count())

def get_graph_vertices_edges(TFIDF_vectors_path,tag_count):
    vectors = np.load(TFIDF_vectors_path)
    with open('tag_id_name.json', "r") as file:
        tag_name_map = json.load(file)
    tag_ids = list(vectors.keys())[:tag_count]
    tag_vectors = list(vectors.values())[:tag_count]
    vertices = [(tag_id,tag_name_map[tag_id]) for tag_id in tag_ids]
   
    similarities = cosine_similarity(tag_vectors,tag_vectors)
    edges = []

    for i in tqdm(range(len(tag_ids))):
        for j in range(len(tag_ids)):
            edges.append((tag_ids[i],tag_ids[j],float(similarities[i][j])))

    return vertices,edges

def community_detection(spark_session,TFIDF_vectors_path,tag_count=2000):
    vertices,edges = get_graph_vertices_edges(TFIDF_vectors_path,tag_count)
    run_community_detection(spark_session.createDataFrame(vertices,["id"]),spark_session.createDataFrame(edges,["src", "dst", "weight"]))

def inference_TFIDF(tag_name,vector_path):
    vector_map = np.load(vector_path)
    with open('tag_name_id.json', "r") as file:
        tag_id_map = json.load(file)
    with open('tag_id_name.json', "r") as file:
        tag_name_map = json.load(file)
   
    id = tag_id_map[tag_name]
    query_vector = vector_map[id].reshape(1, -1)

    similarities = {}
    for key, value in vector_map.items():
        similarity = cosine_similarity(query_vector, value.reshape(1, -1))[0][0]
        similarities[key] = similarity

    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    for idx, (id, similarity) in enumerate(sorted_similarities[:20], start=1):
        print(f"Rank {idx}: tag = {tag_name_map[id]}, Similarity = {similarity:.4f}")
    return sorted_similarities

def prepare_TFIDF_vectors(spark_session,save_path,feature_size=1000):
    print("Here")
    # Build the dataset
    tag_post_df, post_text_df_raw = build_dataset(spark_session)

    # preprocess tag data
    tag_post_df = tag_post_df.withColumn(
    "post_ids_array",
    F.split(F.col("post_ids"), ",").cast("array<string>")
    ).drop('post_ids')

    # Preprocess the post text data 
#     textPreprocessor = TextPreprocessor(post_text_df_raw)
#     post_text_df = textPreprocessor.preprocess_text()
#     post_text_df = post_text_df.filter(post_text_df["score"] >= 1)
#     post_text_df = post_text_df.filter(
#         (trim(col("title")).isNotNull()) & (trim(col("title")) != "") & 
#         (trim(col("body")).isNotNull()) & (trim(col("body")) != "")
#     )

    # convert to pandas dataframe
    tag_post_df_pandas = tag_post_df.toPandas()
    post_text_df_pandas = post_text_df.toPandas()

    tag_ids = []
    tag_names = []
    tag_post_ids = []

    for index,row in tag_post_df_pandas.iterrows():
        tag_ids.append(row['tag_id'])
        tag_names.append(row['tag_name'])
        tag_post_ids.append(row['post_ids_array'])

    body_texts = []
    post_ids = []
    for index,row in post_text_df_pandas.iterrows():
        body_texts.append(row['body'])
        post_ids.append(row['post_id'])


    tag_name_map = {str(tag_id):tag_name for tag_id,tag_name in zip(tag_ids,tag_names)}
    tag_id_map = {tag_name:str(tag_id) for tag_id,tag_name in zip(tag_ids,tag_names)}
    with open('tag_name_id.json', 'w') as file:
        json.dump(tag_id_map,file)
    with open('tag_id_name.json', 'w') as file:
        json.dump(tag_name_map,file)

    # Train TF-IDF vectorizer (1000 features)
    vectorizer_body = TfidfVectorizer(
    analyzer = 'word', 
    strip_accents = None, 
    encoding = 'utf-8', 
    preprocessor=None, 
    max_features=feature_size)

    vectors = vectorizer_body.fit_transform(body_texts).toarray()
    # post id to vector map
    post_id_vector_map = {post_id: vectors[index] for index,post_id in enumerate(post_ids)}

    # tag id to vector map
    tag_vector_map = {}
    for index,tag_id in enumerate(tag_ids):
        tag_posts = tag_post_ids[index]
        tag_posts_vectors = []
        for post_id in tag_posts:
            if post_id in post_ids: tag_posts_vectors.append(post_id_vector_map[post_id])
        
        tag_vector_map[str(tag_id)] = np.mean(np.array(tag_posts_vectors),axis=0) if len(tag_posts_vectors)>0 else np.zeros(feature_size)

    np.savez(save_path, **tag_vector_map)



# In[ ]:


# graphframes_jar_path = "gs://532-dataset/notebooks/jupyter/532/graphframes-0.8.2-spark3.0-s_2.12.jar"


# In[ ]:


# from pyspark.sql import SparkSession

# # Ensure the session is fresh by restarting the kernel or stopping the current Spark session
# # spark_session = SparkSession.builder \
# #     .appName("GraphFramesExample") \
# #     .config("spark.jars", "gs://532-dataset/notebooks/jupyter/532/graphframes-0.8.2-spark3.0-s_2.12.jar") \
# #     .config("spark.hadoop.fs.defaultFS", "gs://532-dataset") \
# #     .config("spark.hadoop.google.cloud.auth.service.account.enable", "true") \
# #     .config("spark.hadoop.google.cloud.auth.service.account.json.keyfile", "<PATH_TO_YOUR_SERVICE_ACCOUNT_JSON>") \
# #     .getOrCreate()

# jars = spark_session.sparkContext.getConf().get("spark.jars")
# print("JARs added to the environment:", jars)
# fs_default = spark_session.sparkContext.getConf().get("spark.hadoop.fs.defaultFS")
# print("Default Filesystem:", fs_default)


# In[ ]:


get_ipython().system('gsutil ls gs://532-dataset/notebooks/jupyter/532/')


# In[ ]:


# # # Stop the current Spark session
# # spark_session.stop()

# # Reconfigure and start a new Spark session
# from pyspark.sql import SparkSession

# # spark_session = SparkSession.builder \
# #     .appName("GraphFramesExample") \
# #     .config("spark.jars", "gs://532-dataset/notebooks/jupyter/532/graphframes-0.8.2-spark3.0-s_2.12.jar") \
# #     .config("spark.hadoop.fs.defaultFS", "gs://532-dataset") \
# #     .getOrCreate()

# # Print configurations to verify
# jars = spark_session.sparkContext.getConf().get("spark.jars")
# print("JARs added to the environment:", jars)
# fs_default = spark_session.sparkContext.getConf().get("spark.hadoop.fs.defaultFS")
# print("Default Filesystem:", fs_default)

# # Set checkpoint directory
# spark_session.sparkContext.setCheckpointDir("gs://532-dataset/notebooks/jupyter/532/spark-checkpoints")


# In[ ]:


# spark_session = SparkSession.builder \
#     .appName("GraphFramesExample") \
#     .config("spark.jars", "gs://532-dataset/notebooks/jupyter/532/graphframes-0.8.2-spark3.0-s_2.12.jar") \
#     .config("spark.hadoop.fs.defaultFS", "gs://532-dataset") \
#     .getOrCreate()
# jars = spark.sparkContext.getConf().get("spark.jars", "")
# print("JARs added to the environment:", jars)
# fs_default = spark_session.sparkContext.getConf().get("spark.hadoop.fs.defaultFS", "Not Configured")
# print("Default Filesystem:", fs_default)
# spark_session.sparkContext.setCheckpointDir("gs://532-dataset/notebooks/jupyter/532/spark-checkpoints")
# jars = spark_session.sparkContext.getConf().get("spark.jars", "")
# print("JARs added to the environment:", jars)

vectors_save_path = 'vectors.npz'
# # Second parameter is the number of TF-IDF features
# prepare_TFIDherefore the checkpoint directory must not be on the local filesystem. Directory '../spark-checkpoints' appears to be on the local filesystem.F_vectors(spark_session,vectors_save_path,1000)
    # sample inference - currently runs only for tags in training data


# In[ ]:


inference_TFIDF('github',vectors_save_path)
# Second parameter is the number of tags over which community detection is run.
community_detection(spark_session,vectors_save_path,20)

spark_session.stop()

