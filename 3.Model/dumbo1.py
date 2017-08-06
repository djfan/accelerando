
from pyspark.mllib.recommendation import *
from pyspark.sql import SQLContext
from pyspark.sql.types import *
import pandas as pd

def eval(model, al, train, valid, rank):
    from pyspark.mllib.recommendation import *
    from pyspark.sql import SQLContext
    from pyspark.sql.types import *
    import pandas as pd
    validMap = valid.map(lambda x:(x[0],x[1])).groupByKey().collectAsMap()
    trainMap = train.map(lambda x:(x[0],x[1])).groupByKey().collectAsMap() 
    alSongs = al.map(lambda x:(x[1])).collect()
    validUserScores = []
    for user in validMap.keys():
        userTrainSongs = trainMap.get(user) 
        try:
            userNotTrainSongs = list(set(alSongs).difference(set(userTrainSongs)))
        except:
            continue
        userNotTrainSongsRDD = sc.parallelize([(user,x) for x in userNotTrainSongs])
        prediction = model.predictAll(userNotTrainSongsRDD).map(lambda x: (x[2], x[1])).sortByKey(False).map(lambda x: x[1])
        userValidSongs = validMap.get(user)
        userValidSongsCounts = len(userValidSongs)
        validUserScores.append(1.0 * len(set(prediction.take(userValidSongsCounts)).intersection(set(userValidSongs))) / userValidSongsCounts)
    validUserMeanScore =  1.0 * sum(validUserScores) / len(validUserScores)
    print "Evaluation Score For Rank {} is {}".format(rank, validUserMeanScore)
    return validUserMeanScore
    

def bestRanking(allData, trainData, validationData, ranks = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]):
    from pyspark.mllib.recommendation import *
    from pyspark.sql import SQLContext
    from pyspark.sql.types import *
    import pandas as pd
    maxScore = 0
    print("ALS.trainImplicit")
    for r in ranks:
        Model = ALS.trainImplicit(ratings=trainData,alpha=0.01,blocks=10,iterations=5,
                                  lambda_=0.01,nonnegative=False,rank=r,seed=826)
        score = eval(Model, allData, trainData, validationData, r)
        if score > maxScore:
            maxScore = score
            bestRank = r
    return bestRank
    
def matchSong(targets, df):
    targets = map(lambda x: str(x[1]), targets)
    res = []
    for r in df:
        if str(r[0]) in targets:
            print "Song: {}\nArtist: {}".format(r[2].encode('utf-8'),r[3].encode('utf-8'))
            res.append((r[2].encode('utf-8'),r[3].encode('utf-8')))
    return res



if __name__ == '__main__':
    from pyspark.mllib.recommendation import *
    from pyspark.sql import SQLContext
    from pyspark.sql.types import *
    import pandas as pd
    from pyspark import SparkContext
    import numpy as np
    
    sc = SparkContext(appName="mrs")
    path1 = 'mrs/id_train_triplets.csv'
    path2 = 'mrs/SongID_int.csv'
    path3 = 'mrs/t.csv'

    musicData = sc.textFile(path1, use_unicode=False).map(lambda x: x.split(',')).map(lambda x: (int(x[3]), int(x[4]), int(x[2])))
    trainData, validationData, testData = musicData.randomSplit([0.4, 0.4, 0.2], 826)
    trainData.cache()
    validationData.cache()
    testData.cache()

    #user = 8

    bestRank_1 = bestRanking(musicData, trainData, validationData, ranks = [5, 10, 15, 20])
    bestModel_1 = ALS.trainImplicit(musicData, rank=bestRank_1, seed=826, blocks=10)

    sqlContext = SQLContext(sc)
    songID = sqlContext.read.format("com.databricks.spark.csv").options(header='true').load(path2)
    songID.registerTempTable("songID")
    songName = sqlContext.read.format("com.databricks.spark.csv").options(header='true').load(path3)
    songName.registerTempTable("songName")

    df = sqlContext.sql('select a.sid, a.song, b.artist, b.title from songID a left join songName b on a.song = b.songid').collect()
    #test = bestModel_1.recommendProducts(8,3)
    #print test
    resAll = {}
    userMeta = sc.textFile(path1, use_unicode=False).map(lambda x: x.split(',')).map(lambda x: (int(x[3]), int(x[4]), int(x[2])))
    users = list(np.unique(userMeta.map(lambda x: x[0]).collect()))
    for user in users:
        try:
            top3_1 = bestModel_1.recommendProducts(user,3)
            res = matchSong(top3_1, df)
        except:
            res = [(None, None), (None, None), (None,None)]
        resAll[user] = tuple(res)
    sc.parallelize(resAll.items()).saveAsTextFile('mrs1_out')
    print 'end!!!!!!!!!!!!!!!!!!!!!!!!!'
