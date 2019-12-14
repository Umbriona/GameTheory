import numpy as np
from Strategies import TitFTatAgent, TitF2tatAgent, RandomChoiceAgent, AlwaysDefectAgent, Neural200Agent

#Function to generate a population of players {scope not updated dependent on Strategies}
def initPlayersRand(nPlayers, p, action_space):
    N = 2
    a = np.array([1, 2, 3, 4, 5])
    arr = np.random.choice(a, nPlayers, p=p)
    list_p = []
    for i in range(nPlayers):
        if arr[i] == 1:
            p = TitFtatAgent('player ' + str(i))
            list_p.append(p)
        elif arr[i] == 2:
            p = TitF2tatAgent('player ' + str(i))
            list_p.append(p)
        elif arr[i] == 3:
            p = AlwaysDefectAgent('player ' + str(i))
            list_p.append(p)
        elif arr[i] == 4:
            p = RandomChoiceAgent('player ' + str(i))
            list_p.append(p)
        elif arr[i] == 5:
            p = Neural200Agent('player ' + str(i), action_space)
            list_p.append(p)
    return list_p, arr

# Function to generate a list of plays for a player population tournamnet
def getTournamentListNotSelf(players):
    tmp = sum(range(players))
    arr = np.zeros([tmp,2])
    count = 0
    for i in range(players-1):
        for j in range(i+1,players):
            arr[count,0] = i
            arr[count,1] = j
            count += 1
    return arr.astype(int)
def getTournamentListSelf(players):
    tmp = sum(range(players))+players
    arr = np.zeros([tmp,2],type)
    count = 0
    for i in range(players):
        for j in range(i,players):
            arr[count,0] = i
            arr[count,1] = j
            count += 1
    return arr.astype(int)