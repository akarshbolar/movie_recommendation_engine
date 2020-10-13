import numpy as np
import pandas as pd
import random
from sklearn.metrics.pairwise import pairwise_distances

l = [line.rstrip('\n').split('\t')[1] for line in open('user_item.txt')]
new_l = []
for i in l:
    if i not in new_l:
        new_l.append(i)
l = new_l

print(len(l))
def gen():
        #This was used for mapping user id to a particular movie name...
        hs = open("user_item.txt","w")
        
        dt = [i.rstrip('\n') for i in open('data.txt')]
        ndt = [j.split('\t') for j in dt]
        for i in ndt:
            i[1] = l[int(i[1])-1]
        for i in ndt:
            tmp = '\t'.join(i)
            #print(tmp)
            hs.write(tmp+'\n')
        hs.close()
#gen()
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

def addUserRating():
    print("ADD NEW USER RATING")
    global l
    rating = []
    print('Enter User Number')
    rating.append(input())
    print("Enter Movie Name")
    i = input()
    if i not in l:
        l.append(i)
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



header = ['user_id', 'movie_id', 'rating', 'timestamp']
df = pd.read_csv('data.txt', sep='\t', names=header)

n_users = df.user_id.unique().shape[0]
n_movies = df.movie_id.unique().shape[0]
print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_movies))


user_item_matrix = np.zeros((n_users, n_movies))
for line in df.itertuples():
    user_item_matrix[line[1]-1, line[2]-1] = line[3]


'''
user_similarity_matrix = pairwise_distances(user_item_matrix, metric='cosine')
print(user_similarity_matrix)
'''

def user_similarity(user_item_matrix, kind='user', epsilon=1e-9):
    # epsilon -> small number for handling divide-by-zero errors
    sim = user_item_matrix.dot(user_item_matrix.T) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

def predict(user_item_matrix, user_similarity_matrix, kind='user'):
    return user_similarity_matrix.dot(user_item_matrix) / np.array([np.abs(user_similarity_matrix).sum(axis=1)]).T

def predict_topk(ratings, similarity, kind='user', k=40):
    pred = np.zeros(ratings.shape)
    for i in range(ratings.shape[0]):
        top_k_users = [np.argsort(similarity[:,i])[:-k-1:-1]]
        for j in range(ratings.shape[1]):
            pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users]) 
            pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
    return pred

#Takes prediction matrix, userID for which to recommend new items, and the number of items to be recommended as arguments
def recommend_movies(predictions, userID, number):
    # userID = userID -1
    global user_item_matrix
    
    most_similar=0.0
    sim_user=0
    for j in range(n_users):
        if(userID!=j and user_similarity_matrix[userID][j]>most_similar):
            most_similar=user_similarity_matrix[userID][j]
            sim_user=j
    sim_user-=1
    #print(sim_user)
    userID = userID -1
    #print(user_item_matrix[sim_user][1690],"vvvgg")
    #predictions[userID][len(predictions[userID])-1] = random.uniform(3.5,5.0)
    movies_of_similar_user=[]
    movies = [] #A list containing the best itemIDs
    ratings = [] #A list containing the ratings of the top itemIDs for that particular user
    for i in range(len(user_item_matrix[sim_user])):
        if(user_item_matrix[userID][i]==0.0 and user_item_matrix[sim_user][i]!=0.0):
            movies_of_similar_user.append((user_item_matrix[sim_user][i],i))
    movies_of_similar_user.sort(reverse=True)
    topnumberrecommendedmovies=[]
    for i in range(number):
        #print(movies_of_similar_user[i])
        topnumberrecommendedmovies.append(movies_of_similar_user[i][1])
        ratings.append(movies_of_similar_user[i][0])
    user_ratings = list(predictions[userID])
    '''while((number > 0) and (len(user_ratings) > 0)):

        max_rating_index = user_ratings.index(max(user_ratings))
        ratings.append(user_ratings[max_rating_index])
        user_ratings.pop(max_rating_index)
        movies.append(max_rating_index)
        number -= 1'''
    #return ratings, movies
    return ratings, topnumberrecommendedmovies

user_similarity_matrix = user_similarity(user_item_matrix)
#print(user_similarity_matrix)

most_similar=0.0
sim_users=[0,0]
for i in range(n_users):
    for j in range(n_users):
        if(i!=j and user_similarity_matrix[i][j]>most_similar):
            most_similar=user_similarity_matrix[i][j]
            sim_users[0]=i
            sim_users[1]=j

predictions = predict_topk(user_item_matrix, user_similarity_matrix)
print(predictions)
while(1):
    s=int(input("do u want to add a user rating\n1.yes\n2.no\n"))
    if(s==1):
        addUserRating()
    else:
        pass

    print("Enter user ID to know the recommendations\n")
    ratings, movies = recommend_movies(predictions, int(input()), 7)
    for i in range(len(ratings)):
        print(str(ratings[i]) + " " + l[movies[i]])
