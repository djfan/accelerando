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

def matchArtist(targets, df):
    targets = map(lambda x: str(x[1]), targets)
    res = []
    for t in targets:
        name = df[df['aid'] == int(t)].artist.values[0]
        name = unicode(name).encode('utf-8')
        print "Artist: {}".format(name)
        res.append(name)
    print '\n'
    return res



if __name__ == '__main__':
    from pyspark.mllib.recommendation import *
    from pyspark.sql import SQLContext, HiveContext
    from pyspark.sql.types import *
    import pandas as pd
    from pyspark import SparkContext
    import numpy as np
    
    sc = SparkContext(appName="mrs2")
    path1 = 'mrs/id_train_triplets.csv'
    path2 = 'mrs/SongID_int.csv'
    path3 = 'mrs/t.csv'
    musicData = sc.textFile(path1, use_unicode=False).map(lambda x: x.split(',')).map(lambda x: (int(x[3]), int(x[4]), int(x[2])))    
    
    sqlContext = SQLContext(sc)
    songID = sqlContext.read.format("com.databricks.spark.csv").options(header='true').load(path2)
    songID.registerTempTable("songID")
    songName = sqlContext.read.format("com.databricks.spark.csv").options(header='true').load(path3)
    songName.registerTempTable("songName")

    df = sqlContext.sql('select a.sid, a.song, b.artist, b.title from songID a left join songName b on a.song = b.songid').collect()
    # SparkSQL conversion
    df2 = sqlContext.createDataFrame(df)
    df2.registerTempTable("songMeta")
    df2.printSchema()
    schema = StructType([
        StructField("uid", IntegerType(), True),
        StructField("sid", StringType(), True),
        StructField("counts", IntegerType(), True)])
    df3 = sqlContext.createDataFrame(musicData, schema)
    df3.registerTempTable("musicData")
    df3.printSchema()
    df4 = sqlContext.sql("select a.uid, b.title from musicData a left join songMeta b on a.sid = b.sid")
    df4.registerTempTable("UserArtist")
    df5 = sqlContext.sql('select uid, title, count(*) from UserArtist group by uid, title').collect()
    schema = StructType([
        StructField("uid",IntegerType(),True),
        StructField('artist',StringType(),True),
        StructField('counts',LongType(),True)])
    df5 = sqlContext.createDataFrame(df5, schema)
    df5.printSchema()
    df5.registerTempTable("Artist")

    df51 = df5.toPandas()
    artList = {}
    for n, a in enumerate(df51.artist.unique()):
        artList[a] = artList.get(a, 0) + (n+1)
    indList = []
    for a in df51.artist:
        ind = artList[a]
        indList.append(ind)
    df51['aid'] = indList
    schema = StructType([
        StructField("uid", ShortType(), True),
        StructField("artist", StringType(), True),
        StructField("counts", ShortType(), True),
        StructField("aid", ShortType(), True)])

    
    df52 = sqlContext.createDataFrame(df51, schema)
    df52.registerTempTable("ArtistID")
    df52.printSchema()

    df7 = sqlContext.sql("select a.uid, a.aid, a.counts as counts from ArtistID a")
    df7.printSchema()
    df7_rdd = df7.rdd.map(tuple)

    # cache
    trainData, validationData, testData = df7_rdd.randomSplit([0.4, 0.4, 0.3], 826)
    trainData.cache()
    validationData.cache()
    testData.cache()

    bestRank_1 = bestRanking(df7_rdd, trainData, validationData, ranks = [5, 10, 15, 20])


    resAll = {}
    userMeta = sc.textFile(path1, use_unicode=False).map(lambda x: x.split(',')).map(lambda x: (int(x[3]), int(x[4]), int(x[2])))
    users = list(np.unique(userMeta.map(lambda x: x[0]).collect()))

    bestModel_artist = ALS.trainImplicit(df7_rdd, rank=bestRank_1, seed=826, blocks=10) 
    for user in users:
        top3_artist = bestModel_artist.recommendProducts(user,3)
        res = matchArtist(top3_artist, df52.toPandas())
        resAll[user] = tuple(list(np.unique(res)))
    df = sc.parallelize(resAll.items()).saveAsTextFile('mrs2_out')
    print 'end!!!!!!!!!!!!!!!!!!!!!!!!!'

