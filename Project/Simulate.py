import numpy as np
import pandas as pd
import os
import Strategies as strat
import inspect
import Games
import Tournament as tour

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
        if '.h5' in i:
            tmp = strat.Neural200Agent(name = i, actionSpace = 2)
            tmp.loadModel('model')
            list_neural_agents.append(tmp)
    return list_neural_agents

def play(list_of_players, rng, v=False): 
    pathData = os.path.join('data','simulations')
    game = Games.PrisonersDilemma('Joshua')
    action_space = game.actionSpace
    
    L = len(list_of_players)
    avgScoreM = np.zeros([L,L])     
    for i in range(30):
        if v:
            print(i)
        for k in list_of_players:
            if k.name[:6] == 'Neural':
                k.prepThread(len(list_of_players))
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
        df[list_of_players[i].name] = avgScoreM[i,:] /30
    df = pd.DataFrame(df)
    df.to_csv(os.path.join(pathData,  'Matrix_avgPlayers.csv'))


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    pathData = os.path.join('data','simulations')
    modelPath = 'model'
    
    list_neural_models = os.listdir(modelPath)
    
    strat_agent_list = print_classes()
    neural_agent_list =  generateAgent(list_neural_models)
    list_of_players = neural_agent_list + strat_agent_list
    print('playing tournament with' )
    for i in list_of_players:
        print(i.name)
    
    play(list_of_players, 0.04,True)
                                  
if __name__ == '__main__':
    main()