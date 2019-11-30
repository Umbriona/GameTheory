import os
import numpy
import argparse
from Arguments import arguments
import inspect


import Strategies as strat
import Games
import Tournament as tour


parser = argparse.ArgumentParser(description='IMSAI 8086')
parser = arguments(parser)
args = parser.parse_args()




def main():
    
    #Initiate game
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    game = Games.PrisonersDilemma('Joshua')
    action_space = game.actionSpace
    
    #Initiate players 
    #for name, obj in inspect.getmembers(Games):
     #   if inspect.isclass(obj): #and name[-5:] == 'Agent':
      #      print (name)
    list_of_players, arr = tour.init_players(20, [0.2, 0.4, 0.2, 0.0, 0.2], action_space)
    
    for i in range(10000):
        print(i)
        game.tournament(list_of_players, 200,0.04)

        for j in range(arr.size):
            if arr[j] == 5:
                list_of_players[j].train()
                list_of_players[j].learning_rate *= 0.95
                
                
    max_avg = 0
    for k in range(len(list_of_players)):    
        if arr[k] == 1:
            avg = 0
            s = 0
            for i in range(len(list_of_players[k].lastScore)):
                s += sum(list_of_players[k].lastScore[i])/200
                avg = (s/len(list_of_players[k].lastScore))
            if max_avg < avg:
                max_avg = avg
                arg = k
    print(max_avg)
    print(arg)


if __name__ == "__main__":
    main()