import numpy as np
import nltk
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs 
from sklearn.metrics import silhouette_score 
from sklearn import metrics
from sklearn import cluster
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
import json
import fasttext
import fasttext.util
import math
from gensim.models.wrappers import FastText

from langdetect import detect


# Make Array Of Title's
def getNewsTitlesFromJson(jsonData):
    ArrayOfSentence = []
    for data in jsonData:
        splittedSentence = data["Title"]
        ArrayOfSentence.append(splittedSentence)
    return ArrayOfSentence

# Generate VectorForms as NumPy Array Using FastText Model
def getVectorsFromFastText(titleList,language):
    vectorValues = []
    if(language == "ta"):
        model = fasttext.load_model("D:/Models/cc.ta.300.bin")
       
    elif(language == "en"):
        model = fasttext.load_model("D:/Models/cc.en.300.bin")

    for title in titleList:
       # print(title)
        a = model.get_sentence_vector(title)
        #print(a)
        vectorValues.append(a)

    numpyVectorArray = np.array(vectorValues)
    return numpyVectorArray

# Find the Optimal Number Of Cluster using SilhouetteMaxScore Method
def findSilhouetteMaxScore(vectorArray):
    silhouetteScore = []
    for n_clusters in range(2,len(vectorArray)): 
        cluster = KMeans(n_clusters = n_clusters) 
        cluster_labels = cluster.fit_predict(vectorArray)
        silhouette_avg = silhouette_score(vectorArray, cluster_labels)
        silhouetteScore.append(silhouette_avg)
   # print(silhouetteScore)
    maxpos = silhouetteScore.index(max(silhouetteScore)) 
    print(maxpos+2)
    return maxpos+2

# Cluster the NewsArticle BY K-Means
def clusterArticleByKMeans(clusterNumber,vectors,newsArticleJson):

    clf = KMeans(n_clusters = clusterNumber, init = 'k-means++')
    labels = clf.fit_predict(vectors)

    for index, newsArticle in enumerate(newsArticleJson):
        labelValue = labels[index] 
        newsArticle["ClusterId"] = int(labelValue)+1
    return sorted(newsArticleJson, key = lambda i: (i['ClusterId']))

def detectLanguage(datas):
    for x in datas:
            language = detect(x["Title"])
            x["Language"] = language
    return datas

# Api endPoint
import flask
from flask import request, jsonify, Response

app = flask.Flask(__name__)
app.config["DEBUG"] = True

# A route to return all of the available entries in our catalog.
@app.route('/api/v1/getcluster', methods=['POST'])
def cluster_all():
    req_data = request.get_json()
    language = req_data['language']
    jsonTitles = req_data['titles']
    newsTitles = getNewsTitlesFromJson(jsonTitles)
    newsVectors = getVectorsFromFastText(newsTitles,language)
    noOfClusters = findSilhouetteMaxScore(newsVectors)
    clusteredJson = clusterArticleByKMeans(noOfClusters,newsVectors,jsonTitles)
    clusteredJsonResult = json.dumps(clusteredJson,ensure_ascii=False,indent=4)
    return clusteredJsonResult

@app.route('/api/v1/detectlanguage', methods=['POST'])
def lanuageDetect_all():
    req_data = request.get_json()
    result_data = detectLanguage(req_data)
    return  jsonify(result_data)

if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=80)
    app.run()