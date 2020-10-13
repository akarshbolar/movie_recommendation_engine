
l = [line.rstrip('\n').split('|')[1] for line in open('item.txt')]

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
    h.write('\t'.join(rating))
    h.close()

def makeFileReady():
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
makeFileReady()
