import mysql.connector
import datetime
from datetime import datetime, timedelta
import json

today = datetime.now()
yesterday = datetime.today() - timedelta(days=1)

today = today.strftime("%Y-%m-%d 18:30:00")
yesterday = yesterday.strftime("%Y-%m-%d 18:30:00")

mydb = mysql.connector.connect(host="52.188.110.40",port=3307,user="user",password="tWXg5p8FK6JpvICDcYQ%fppxbJa",database="newstuckstage")
mycursor = mydb.cursor(buffered=True)
mycursor = mydb.cursor(prepared=True)
mycursor = mydb.cursor()
query = """SELECT FeedItemId,Title FROM FeedItems WHERE PublishDate >= (%s) and PublishDate <= (%s)"""
dateValues = (yesterday,today)
mycursor.execute(query,dateValues)
resultsFromDatabase = mycursor.fetchall()

# for i in resultsFromDatabase:
#     print(i)

with open('DBResults.json', 'w', encoding='utf-8') as f:
    json.dump(resultsFromDatabase, f, ensure_ascii=False, indent=4)

