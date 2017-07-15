import pandas as pd
data = pd.read_table("./train_triplets.txt", header=None, nrows=1000)
data.columns = ['user', 'song', 'counts']
userID = pd.DataFrame({'user':data.user.unique(), 'uid':range(data.user.unique().shape[0])})
songID = pd.DataFrame({'song':data.song.unique(), 'sid':range(data.song.unique().shape[0])})
# print userID.shape
# print songID.shape
data2 = data.copy()
data2 = pd.merge(data2, userID, on='user')
data2 = pd.merge(data2, songID, on='song')
data2.to_csv("./id_train_triplets.csv", index=False, header=False)