import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances

train_data_matrix=None
test_data_matrix=None
user_similarity=None
l = [line.rstrip('\n').split('|')[1] for line in open('item.txt')]

def gen():
    pass
    ''' This was used for mapping user id to a particular movie name...
        hs = open("user_item.txt","w")
        
        dt = [i.rstrip('\n') for i in open('ml-100k/u.data')]
        ndt = [j.split('\t') for j in dt]
        for i in ndt:
        i[1] = l[int(i[1])-1]
        for i in ndt:
        tmp = '\t'.join(i)
        #print(tmp)
        hs.write(tmp+'\n')
        hs.close() '''

def addUserRating():
    global l
    rating = []
    print('Enter User Number\n')
    rating.append(input())
    print("Enter Movie Name")
    i = input()
    rating.append(i)
    if(i not in l):
        l.append(i)
    print("Enter User Rating")
    rating.append(input())
    print("Enter Timestamp")
    rating.append(input())
    h = open("user_item.txt","a")
    h.write('\t'.join(rating)+'\n')
    h.close()
    makeFileReady()

def makeFileReady():
    global l
    data = open("data.txt","w")
    dt = [i.rstrip('\n') for i in open('user_item.txt')]
    ndt = [j.split('\t') for j in dt]
    for i in ndt:
        i[1] = str(l.index(i[1])+1)
    for i in ndt:
        tmp = '\t'.join(i)
        #print(tmp)
        data.write(tmp+'\n')
    data.close()

def mainpredictor():
    global train_data_matrix,test_data_matrix,user_similarity
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv('data.txt', sep='\t', names=header)

    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]
    print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

    train_data, test_data = train_test_split(df, test_size=0.25)

    train_data_matrix = np.zeros((n_users, n_items))
    for line in train_data.itertuples():
        train_data_matrix[line[1]-1, line[2]-1] = line[3]

    test_data_matrix = np.zeros((n_users, n_items))
    for line in test_data.itertuples():
        test_data_matrix[line[1]-1, line[2]-1] = line[3]

    print("Training data: ")
    print(train_data_matrix)
    print("\nTest data: ")
    print(test_data_matrix)

    user_similarity = pairwise_distances(train_data_matrix, metric='cosine')

    print("Similarity Matrix", end = "\n")
    print(user_similarity)

    x = np.asarray([[1,2,3],[4,5,6]])
    y = np.asarray([1,2])
    #print(x - y[:, np.newaxis])

    #print(list(train_data_matrix[0]))
    #print(train_data_matrix.mean(axis=1)[0])
    #print((train_data_matrix - train_data_matrix.mean(axis=1)[:, np.newaxis])[0])


def predict(ratings, similarity, type='user'):

    mean_user_rating = ratings.mean(axis=1)
    ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
    pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    return pred


def predictMovie(userId):
    global user_prediction
    user = list(user_prediction[userId])
    print(user)
    newUser = user[::]
    user.sort(reverse=True)
    top5 = user[:5]

    top5index = []
    for i in range(len(top5)):
        #print(newUser.index(top5[i]))
        top5index.append(newUser.index(top5[i]))
    print(top5index)
    global l
    top5MovieNames = []
    for i in top5index:
        top5MovieNames.append(l[i])
    print(top5MovieNames)

mainpredictor()
user_prediction = predict(train_data_matrix, user_similarity, type='user')
userid = int(input("please enter user id to get its predictions - "))
predictMovie(userid)



