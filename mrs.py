from pyspark.mllib.recommendation import *

def eval(model, al, train, valid, rank):
    '''
    al: all
    trian: trianing
    valid: validation
    '''
    validMap = valid.map(lambda x:(x[0],x[1])).groupByKey().collectAsMap()
    trainMap = train.map(lambda x:(x[0],x[1])).groupByKey().collectAsMap() 
    alSongs = al.map(lambda x:(x[1])).collect()
    validUserScores = []
    for user in validMap.keys():
        userTrainSongs = trainMap.get(user) 
        userNotTrainSongs = list(set(alSongs).difference(set(userTrainSongs)))
        userNotTrainSongsRDD = sc.parallelize([(user,x) for x in userNotTrainSongs])
        userValidSongs = validMap.get(user)
        userValidSongsCounts = len(userValidSongs)
        validUserScores.append(1.0 * len(set(prediction.take(userValidSongsCounts)).intersection(set(userValidSongs))) / userValidSongsCounts)
    validUserMeanScore =  1.0 * sum(validUserScores) / len(validUserScores)
    print "Evaluation Score For Rank {} is {}".format(rank, validUserMeanScore)

musicData = sc.textFile('./id_train_triplets.csv', use_unicode=False).map(lambda x: x.split(',')).map(lambda x: (int(x[3]), int(x[4]), int(x[2])))

trainData, validData, testData = musicData.randomSplit([0.4, 0.4, 0.3], 13)
trainData.cache()
validationData.cache()
testData.cache()

ranks=[10, 20, 30, 40]
# ALS.trainImplicit
for r in ranks:
    Model = ALS.trainImplicit(trainData, rank=r, seed=826) 
    modelEval(Model, musicData, trainData, validationData, r)
bestModel_1 = ALS.trainImplicit(trainData, rank=10, seed=826) 
modelEval(bestModel_1, musicData, trainData, testData, rank=10)
top3_1 = bestModel_1.recommendProducts(1,3)
print top3_1

# ALS.train
for r in ranks:
    Model = ALS.train(trainData, rank=r, seed=826)
    modelEval(Model, musicData, trainData, validationData, r)
bestModel_2 = ALS.train(trainData, rank=20, seed=826) 
modelEval(bestModel_2, musicData, trainData, testData, rank=20)
top3_2 = bestModel_2.recommendProducts(1,3)
print top3_2