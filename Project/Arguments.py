import os
def arguments(parser):
    
    #-----------------------------------------------------------
    # Arguments to be parsed for the program Simulate.py 
    # Defalut values are based on @sandra server config!
    #
    #
    #-----------------------------------------------------------

    parser.add_argument('--Game_name', type = str, required= False,
                       help = 'What game to simulate ex. PrisonersDilemma  ')
    #parser.add_argument('--')
    return parser

def args(parser):
    
    #-----------------------------------------------------------
    # Arguments to be parsed for the program Simulate.py 
    # Defalut values are based on @sandra server config!
    #
    #
    #-----------------------------------------------------------

    parser.add_argument('--Sim', type = str, required= False,
                       help = 'What game to simulate ex. PrisonersDilemma  ')
    #parser.add_argument('--')
    return parser 