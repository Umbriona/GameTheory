import numpy as np
import pandas as pd
import os
import Strategies as strat
import inspect
import Games
import Tournament as tour
from multiprocessing import Process

import argparse
from Arguments import args

parser = argparse.ArgumentParser(description='IMSAI 8086')
parser = args(parser)
arg = parser.parse_args()

def print_classes():
    obj_list = []
    name_list = []
    count = 0
    for name, obj in inspect.getmembers(strat):
        if name[-5:] == 'Agent' and 'MS' not in name:
            if 'Neural' not in name:
                obj_list.append(obj(name = name))
    return obj_list

def generateAgent(list_models):
    list_neural_agents = []
    for i in list_models:
        if '.h5' in i and 'rng' in i and not  'OpMe' in i:
            tmp = strat.Neural201Agent(name = i, actionSpace = 2)
            tmp.loadModel('model')
            list_neural_agents.append(tmp)
    return list_neural_agents

def countSwitches(arr):
    arr = arr.astype(np.int32)
    count = np.sum(np.bitwise_xor(arr[:-1], arr[1:]))
    return count

def play(list_of_players, rng, v=False): 
    pathData = os.path.join('data','simulations')
    game = Games.PrisonersDilemma('Joshua')
    action_space = game.actionSpace
    
    L = len(list_of_players)
    avgScoreM = np.zeros([L,L])     
    for i in range(5):
        if v:
            print(i)
        for k in list_of_players:
            if k.name[:6] == 'Neural':
                k.prepThread(len(list_of_players), 16)
            else:
                k.clearHistory(len(list_of_players))
        game.tournament(list_of_players, 200,rng, True)
        for j in range(L):
            for k in range(L):
                avgScoreM[j,k] += np.sum(list_of_players[j].lastScore[k,:]) / 200 
            try:
                list_of_players[j].resetState()
            except:
                pass
    #print(avgScoreM/100)
    df = {}
    #arr = np.array(list_avgScore)
    for i in range(avgScoreM.shape[0]):
        df[list_of_players[i].name] = avgScoreM[i,:] /5
    df = pd.DataFrame(df)
    df.to_csv(os.path.join(pathData,  'Matrix_avgPlayers_rng.csv'))

def stability(nReplicates, epochs, gamma, v = False):
    game = Games.PrisonersDilemma('Joshua')
    action_space = game.actionSpace
    pathData = os.path.join('data','simulations')

    
    

    avgScoreM = np.zeros(epochs*nReplicates) 
    avg_op = np.zeros(epochs*nReplicates)
    avg_me = np.zeros(epochs*nReplicates)
    avg_switches_op = np.zeros(epochs*nReplicates)
    avg_switches_me = np.zeros(epochs*nReplicates)
    f = 0
    for i in range(nReplicates):
        player1 = strat.Neural203Agent(name = 'Neural', actionSpace = game.actionSpace)
        player1.gamma = gamma
        player3 = strat.TitF2tatAgent(name = 'TitF2Tat')
        list_of_players = [player1, player3]
        if v:
            print('process id:', os.getpid(), '\tReplicat nr:',i)
        for j in range(epochs*i,epochs*i+epochs):
            for k in list_of_players:

                if k.name == 'Neural':
                    k.prepThread(len(list_of_players)-1+f, 10)
                else:
                    k.clearHistory(len(list_of_players)-1+f)
            game.tournament(list_of_players, 200, 0.05, False)
            avgScoreM[j] = np.sum(list_of_players[0].lastScore) / (200 ) 
            avg_op[j] = np.sum(list_of_players[0].lastOp) / (200 )
            avg_me[j] = np.sum(list_of_players[0].lastMe) / (200 )
            avg_switches_op[j] = countSwitches(list_of_players[0].lastOp[0]) / (200 )
            avg_switches_me[j] = countSwitches(list_of_players[0].lastMe[0]) / (200 )
            for j in list_of_players:
                if j.name == 'Neural' :
                    j.train()
    df = {}
    df['avgScore'] = avgScoreM 
    df['avgOp'] = avg_op
    df['avgMe'] = avg_me
    df['avgSOp'] = avg_switches_op
    df['avgSMe'] = avg_switches_me
    
    df = pd.DataFrame(df)
    df.to_csv(os.path.join(pathData,  'Stability_data_gamma{}'.format(gamma)))
    
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    pathData = os.path.join('data','simulations')
    modelPath = 'model'
    
    if ( arg.Sim == "Play"):
        list_neural_models = os.listdir(modelPath)
        strat_agent_list = print_classes()
        neural_agent_list =  generateAgent(list_neural_models)
        list_of_players = neural_agent_list + strat_agent_list
        print('playing tournament with' )
        for i in list_of_players:
            print(i.name)
        play(list_of_players, 0.0,True)
                                       
    elif(arg.Sim == "Stability"):
        arr_gamma = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        
        list_of_processes = [Process(target = stability, args = (20, 2000, i, True)) for i in arr_gamma]
        thread = 16
        for i in range(len(list_of_processes)//thread + 1):
            for j in range(i*thread ,min(i*thread+thread, len(list_of_processes))):
                list_of_processes[j].start()
            for j in range(i*thread ,min(i*thread+thread, len(list_of_processes))):
                list_of_processes[j].join()
        
    else:
        print("No such simulation as: {}".format(arg.Sim))
                                  
if __name__ == '__main__':
    main()