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

def getNewsArticlesJson():
    file  = open("EnglishArticles.json",encoding="utf8")
    datas = json.load(file)
    return datas

def getNewsTitlesFromJson(jsonData):
    ArrayOfSentence = []
    for data in jsonData:
        splittedSentence = data["Title"]
        ArrayOfSentence.append(splittedSentence)
    return ArrayOfSentence

def getVectorsFromFastText(titleList):
    vectorValues = []
    model = fasttext.load_model("D:/Models/cc.en.300.bin")

    for title in titleList:
        a = model.get_word_vector(title)
        vectorValues.append(a)

    numpyVectorArray = np.array(vectorValues)
    return numpyVectorArray

def findElbowFromVector(vectorArray):
    elbow=[]
    for i in range(1, 10):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 101)
        kmeans.fit(vectorArray)
        elbow.append(kmeans.inertia_)
    elbowBend = 5
    return elbowBend

def findSilhouetteMaxScore(vectorArray):
    silhouetteScore = []
    for n_clusters in range(2,len(vectorArray)): 
        cluster = KMeans(n_clusters = n_clusters) 
        cluster_labels = cluster.fit_predict(vectorArray)
       # print(cluster_labels)  
        silhouette_avg = silhouette_score(vectorArray, cluster_labels)
       # print("SAV", silhouette_avg)
        silhouetteScore.append(silhouette_avg)
       # print("For no of clusters =", n_clusters, " The average silhouette_score is :", silhouette_avg)
    maxpos = silhouetteScore.index(max(silhouetteScore)) 
    print(maxpos+2)
    return maxpos+2
 

def clusterArticleByKMeans(clusterNumber,vectors,newsArticleJson):
    clf = KMeans(n_clusters = clusterNumber, init = 'k-means++')
    labels = clf.fit_predict(vectors)

    for index, newsArticle in enumerate(newsArticleJson):
        labelValue = labels[index]
        newsArticleJson[index]["ClusterId"] = int(labelValue)
    return sorted(newsArticleJson, key = lambda i: (i['ClusterId']))

def writeClsuterdJson(clusteredJson):
    with open('ClusterEnglishNews.json', 'w', encoding='utf-8') as f:
        json.dump(clusteredJson, f, ensure_ascii=False, indent=4)

def calculate_wcss(data):
    wcss = []
    for n in range(2, len(data)):
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(X=data)
        wcss.append(kmeans.inertia_)
    
    #print(wcss)
    return wcss

def optimal_number_of_clusters(wcss):
    x1, y1 = 2, wcss[0]
    x2, y2 = 20, wcss[len(wcss)-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    
    print(distances.index(max(distances)) + 2)
    return distances.index(max(distances)) + 2

newsArticleJson = getNewsArticlesJson()
newsTitles = getNewsTitlesFromJson(newsArticleJson)
newsVectors = getVectorsFromFastText(newsTitles)
#noOfClusters = findElbowFromVector(newsVectors)
noOfClusters = findSilhouetteMaxScore(newsVectors)
#wcss = calculate_wcss(newsVectors)
#onc = optimal_number_of_clusters(wcss)
clusteredJson = clusterArticleByKMeans(noOfClusters,newsVectors,newsArticleJson)
writeClsuterdJson(clusteredJson)




