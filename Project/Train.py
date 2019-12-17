import os
import numpy as np
import pandas as pd
import argparse
from Arguments import arguments
import inspect
from multiprocessing import Process

import Strategies as strat
import Games
import Tournament as tour


parser = argparse.ArgumentParser(description='IMSAI 8086')
parser = arguments(parser)
args = parser.parse_args()

def print_classes():
    obj_list = []
    name_list = []
    count = 0
    for name, obj in inspect.getmembers(strat):
        if name[-5:] == 'Agent' and 'MS' not in name:
            if 'Neural' not in name:
                obj_list.append(obj(name = name))
    return obj_list

def train(list_of_players, rng = 0.0, v = False):
    
    pathData = os.path.join('data','training')
    pathModel = 'model'
    game = Games.PrisonersDilemma('Joshua')
    action_space = game.actionSpace
    
    list_avgScore = []   
    for i in range(1000):
        if v:
            print(i)
        for k in list_of_players:

            if k.name[:6] == 'Neural':
                k.prepThread(len(list_of_players)-1)
            else:
                k.clearHistory(len(list_of_players)-1)
        game.tournament(list_of_players, 200,rng)
        avgScore= []
        for j in list_of_players:
            avgScore.append(np.sum(j.lastScore) / (200 * (len(list_of_players)-1)))
            if j.name[:6] == 'Neural' :
                j.train()
            try:
                j.resetState()
            except:
                pass
        list_avgScore.append(avgScore)

    list_of_players[0].saveModel(path = pathModel)
    df = {}
    arr = np.array(list_avgScore)
    for i in range(arr.shape[1]):
        df[list_of_players[i].name] = arr[:,i] 
    df = pd.DataFrame(df)
    df.to_csv(os.path.join(pathData, list_of_players[0].name +'.csv'))

def main():
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    train_1v1 = False
    #Initiate Players 1 vs 1
    if train_1v1 == True:
        obj_list = print_classes()
        neural_list = [strat.Neural200Agent(name = 'NeuralAgent_'+ obj_list[i].name + '_rng', actionSpace = 2) for i in range(len(obj_list))]
        list_of_players = [[neural_list[i], obj_list[i]] for i in range(len(obj_list))]

        #print(Process(target = train, args = (list_of_players[i], 0.04)))
        list_of_processes = [Process(target = train, args = (list_of_players[i], 0.04)) for i in range(len(list_of_players))]
        thread = 16
        for i in range(len(list_of_processes)//thread):
            for j in range(i*thread ,min(i*thread+thread, len(list_of_processes))):
                list_of_processes[j].start()
            for j in range(i*thread ,min(i*thread+thread, len(list_of_processes))):
                list_of_processes[j].join()
    
    else:
        print('Battle Royal!!!!')
        obj_list = print_classes()
        list_of_players = []
        list_of_players.append(strat.Neural200Agent(name = 'NeuralAgent_BattleRoyal_rng', actionSpace = 2))
        list_of_players += obj_list
        train(list_of_players, 0.04, v = True)
    return 0


if __name__ == "__main__":
    end = main()
    print(end)