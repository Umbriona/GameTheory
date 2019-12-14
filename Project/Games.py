import numpy as np
from multiprocessing import Pool

import Tournament as tour
#---------------------------------Basic clases---------------------------

class Basic():
    def __init__(self, name):
        self.name = name
        self.actionSpace = None
        self.payofMatrix = None
        self.nThreads = 4
        self.nIter = 200
        self.rng = 0.0

#-----------------------------------Games--------------------------------

class PrisonersDilemma(Basic):
    
    def __init__(self, name):
        super(PrisonersDilemma, self).__init__(name)
        self.actionSpace = 2
        self.payofMatrix = np.array([[3, 0],[5, 1]])
        
    def tournament(self, list_p, nIter, rng):
        for i in range(len(list_p)):
            for j in range(i + 1,len(list_p)):
                player_1 = list_p[i]
                player_2 = list_p[j]
                move_player_1 = np.zeros(self.nIter)
                move_player_2 = np.zeros(self.nIter)
                Score_player_1 = np.zeros(self.nIter)
                Score_player_2 = np.zeros(self.nIter)
                for t in range(self.nIter):
                    if player_1.name[:7] =='Neural': 
                        tmp = player_1.chooseAction(move_player_1,move_player_2,t, index = j)
                    else:
                        tmp = player_1.chooseAction(move_player_1,move_player_2,t)

                    if np.random.rand()> self.rng:
                        move_player_1[t] = tmp
                    else: 
                        move_player_1[t] = 1 - tmp

                    if player_2.name[:7] == 'Neural':
                        tmp = player_2.chooseAction(move_player_2,move_player_1,t, index = i)
                    else:
                        tmp = player_2.chooseAction(move_player_2,move_player_1,t)

                    if np.random.rand() > self.rng:
                        move_player_2[t] = tmp
                    else:
                        move_player_2[t] = 1 - tmp
                    if move_player_1[t] == 1 and move_player_2[t] == 1:
                        Score_player_1[t] = self.payofMatrix[0,0]
                        Score_player_2[t] = self.payofMatrix[0,0]
                    if move_player_1[t] == 1 and move_player_2[t] == 0:
                        Score_player_2[t] = self.payofMatrix[1,0]
                        Score_player_1[t] = self.payofMatrix[0,1]
                    if move_player_1[t] == 0 and move_player_2[t] == 1:
                        Score_player_1[t] = self.payofMatrix[1,0]
                        Score_player_2[t] = self.payofMatrix[0,1]
                    if move_player_1[t] == 0 and move_player_2[t] == 0:
                        Score_player_1[t] = self.payofMatrix[1,1]
                        Score_player_2[t] = self.payofMatrix[1,1]
                        
                player_1.lastScore[j] = Score_player_1
                player_2.lastScore[i] = Score_player_2
                player_1.lastMe[j] = move_player_1
                player_1.lastOp[j] = move_player_2
                player_2.lastMe[i] = move_player_2
                player_2.lastOp[i] = move_player_1
        return 0

    def playRound(self, list_map, index):
        player_1 = list_p[0]
        player_2 = list_p[1]
        move_player_1 = np.zeros(self.nIter)
        move_player_2 = np.zeros(self.nIter)
        Score_player_1 = np.zeros(self.nIter)
        Score_player_2 = np.zeros(self.nIter)
        for t in range(self.nIter):
            if player_1.name[:7] =='Neural': 
                tmp = player_1.chooseAction(move_player_1,move_player_2,t, index = index[1])
            else:
                tmp = player_1.chooseAction(move_player_1,move_player_2,t)

            if np.random.rand()> self.rng:
                move_player_1[t] = tmp
            else: 
                move_player_1[t] = 1 - tmp

            if player_2.name[:7] == 'Neural':
                tmp = player_2.chooseAction(move_player_2,move_player_1,t, index = [0])
            else:
                tmp = player_2.chooseAction(move_player_2,move_player_1,t)

            if np.random.rand() > self.rng:
                move_player_2[t] = tmp
            else:
                move_player_2[t] = 1 - tmp
            if move_player_1[t] == 1 and move_player_2[t] == 1:
                Score_player_1[t] = self.payofMatrix[0,0]
                Score_player_2[t] = self.payofMatrix[0,0]
            if move_player_1[t] == 1 and move_player_2[t] == 0:
                Score_player_2[t] = self.payofMatrix[1,0]
                Score_player_1[t] = self.payofMatrix[0,1]
            if move_player_1[t] == 0 and move_player_2[t] == 1:
                Score_player_1[t] = self.payofMatrix[1,0]
                Score_player_2[t] = self.payofMatrix[0,1]
            if move_player_1[t] == 0 and move_player_2[t] == 0:
                Score_player_1[t] = self.payofMatrix[1,1]
                Score_player_2[t] = self.payofMatrix[1,1]
                
        player_1.lastScore[index[1]] = Score_player_1
        player_2.lastScore[index[0]] = Score_player_2
        player_1.lastMe[index[1]] = move_player_1
        player_1.lastOp[index[1]] = move_player_2
        player_2.lastMe[index[0]] = move_player_2
        player_2.lastOp[index[0]] = move_player_1
        return 0
    
    def tournamentThread(self, list_p, nIter, rng):
        for i in list_p:
            if i.name[:7] == 'Neural':
                i.prepThread(len(list_p))

        identifyer = tour.getTournamentListNotSelf(len(list_p))
        list_map = [[list_p[identifyer[i,0]], list_p[identifyer[i,1]],identifyer[i,:]]for i in range(identifyer.shape[0])]

        with Pool(self.nThreads) as p: 
            p.map( self.playRound, list_map)


    
            
            