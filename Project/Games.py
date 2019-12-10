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
                move_1 = np.zeros(nIter)
                move_2 = np.zeros(nIter)
                tempScore_1 = np.zeros(nIter)
                tempScore_2 = np.zeros(nIter)
                for t in range(nIter):
                    tmp = player_1.chooseAction(move_1,move_2,t)
                    if np.random.rand()> rng:
                        move_1[t] = tmp
                    else: 
                        move_1[t] = 1 - tmp
                        
                    tmp = player_2.chooseAction(move_2,move_1,t)
                    if np.random.rand() > rng:
                        move_2[t] = tmp
                    else:
                        move_2[t] = 1 - tmp
                    if move_1[t] == 1 and move_2[t] == 1:
                        tempScore_1[t] += 3
                        tempScore_2[t] += 3
                    if move_1[t] == 1 and move_2[t] == 0:
                        tempScore_2[t] += 5
                    if move_1[t] == 0 and move_2[t] == 1:
                        tempScore_1[t] += 5
                    if move_1[t] == 0 and move_2[t] == 0:
                        tempScore_1[t] += 1
                        tempScore_2[t] += 1
                player_1.lastScore.append(tempScore_1)
                player_2.lastScore.append(tempScore_2)
                player_1.lastMe.append(move_1)
                player_1.lastOp.append(move_2)
                player_2.lastMe.append(move_2)
                player_2.lastOp.append(move_1)
        return 0

    def playRound(self, list_map):
        player_1 = list_p[0]
        player_2 = list_p[1]
        move_player_1 = np.zeros(self.nIter)
        move_player_2 = np.zeros(self.nIter)
        Score_player_1 = np.zeros(self.nIter)
        Score_player_2 = np.zeros(self.nIter)
        for t in range(self.nIter):
                    tmp = player_1.chooseAction(move_player_1,move_player_2,t)
                    if np.random.rand()> self.rng:
                        move_player_1[t] = tmp
                    else: 
                        move_player_1[t] = 1 - tmp
                        
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

        return Score_player_1, Score_player_2, move_player_1, move_players_2, list_map[2]
    
     def tournamentThread(self, list_p, nIter, rng):
            
        identifyer = tour.getTournamentListNotSelf(len(list_p))
        list_map = [[list_p[identifyer[i,0]], list_p[identifyer[i,1]],identifyer[i,:]]for i in range(identifyer.shape[0])]
        with Pool(self.nThreads) as p: 
            Score_player_1, Score_player_2, move_player_1, move_players_2, identifyer = p.map( playRound, list_map)
        for i in range(len(identifyer)):
            list_p[identifyer[i]].

    
            
            