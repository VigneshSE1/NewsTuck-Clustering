import json
from collections import defaultdict

file  = open("ManuallyTamilClustered.json",encoding="utf8")
ManualCluster = json.load(file)

file  = open("ProgrammaticallyTamilClustered.json",encoding="utf8")
PredictedCluster = json.load(file)

def checkNoCluster(clusteredJson):
    maxInCluster = 0
    for item in clusteredJson:
        if(item["ClusterId"] > maxInCluster):
            maxInCluster = item["ClusterId"]
    return maxInCluster

def getClusterList(clusteredJson):
    #noOFcluster = checkNoCluster(clusteredJson)
   # clusterList = [[] for i in range(int(noOFcluster))]

    #for i,cluster in clusteredJson:
        #cluster.append(cluster["ClusterId"])
    groups = defaultdict(list)

    for obj in clusteredJson:
        #print(obj)
        #print('_______________________________________________________________________________')
        groups[obj["ClusterId"]].append(obj["FeedItemId"])
    new_list = groups.values()
    #print(clusterList)
    #return clusterList
    return new_list

noOfClusterInManual = checkNoCluster(ManualCluster)
noOfClusterInPredicted = checkNoCluster(PredictedCluster)

print("No Of Manual Cluster ==> ", noOfClusterInManual)
print("No Of Predicted Cluster ==>", noOfClusterInPredicted)

ManualClusterList = getClusterList(ManualCluster)
PredictedClusterList = getClusterList(PredictedCluster)

print("__________________________________________________________")

print("Manual Cluster List ==> ", ManualClusterList[1] )
print("Predicted Cluster List ==>", PredictedClusterList)