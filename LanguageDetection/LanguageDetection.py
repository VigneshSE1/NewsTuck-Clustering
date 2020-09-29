from langdetect import detect
import json

#datas =[]

def getNewsArticlesJson():
    file  = open("D:/NewsClustering_Python/Demo/DataSet.json",encoding="utf8")
    datas = json.load(file)
    return datas

def writeClsuterdJson(datas):
    #print(datas)
    tamilArticles = []
    englishAticles = []
    for data in datas:
        if(data["language"] == "ta"):
            tamilArticles.append(data)
        if(data["language"] == "en"):
            englishAticles.append(data)

    with open('D:/NewsClustering_Python/Demo/DataSet3.json', 'w', encoding='utf-8') as f:
        json.dump(tamilArticles, f, ensure_ascii=False, indent=4)
   # with open('D:/NewsClustering_Python/Demo/EnglishArticles.json', 'w', encoding='utf-8') as f:
      #  json.dump(englishAticles, f, ensure_ascii=False, indent=4)

datas = getNewsArticlesJson()
for x in datas:
    language = detect(x["Title"])
    x["language"] = language
    
writeClsuterdJson(datas)
