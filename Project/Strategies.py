import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import softmax
from tensorflow.keras.optimizers import Adam

#----------------------------------Parent Classes------------------------------
class Basic():
    
    def __init__(self,name):
        self.name = name
        self.numberOfRounds = 200
        self.lastScore = np.zeros([1,self.numberOfRounds])
        self.lastMe = np.zeros([1,self.numberOfRounds])
        self.lastOp = np.zeros([1,self.numberOfRounds])
        
        
    def chooseAction(self):
        return 1
    
    def clearHistory(self, nPlayers):
        self.lastScore = np.zeros([nPlayers,self.numberOfRounds])
        self.lastMe = np.zeros([nPlayers,self.numberOfRounds])
        self.lastOp = np.zeros([nPlayers,self.numberOfRounds])

class BasicNeuron(Basic):

    def __init__(self,name, actionSpace):
        super(BasicNeuron, self).__init__(name)
        self.actionSpace = actionSpace
        self.learning_rate = 0.001
        self.gamma = 0.8
        self.prob = np.zeros([1,self.numberOfRounds,2])
        self.states = np.zeros([1,self.numberOfRounds,10])
    
    def prepThread(self,nPlayers):
        self.prob = np.zeros([nPlayers,self.numberOfRounds,2])
        self.states = np.zeros([nPlayers,self.numberOfRounds,10])
        self.lastScore = np.zeros([nPlayers,self.numberOfRounds])
        self.lastMe = np.zeros([nPlayers,self.numberOfRounds])
        self.lastOp = np.zeros([nPlayers,self.numberOfRounds])
        
    def makeModel(self):
        model = Sequential()
        for i in range(self.nLayers-1):
            if i == 0:
                model.add(Dense(self.nNeurons[i], input_shape=(self.inputSize,), activation= 'relu', use_bias = True))
            else:
                model.add(Dense(self.nNeurons[i], activation = 'relu', use_bias = True))
        model.add(Dense(self.nNeurons[-1], activation = softmax))
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model
    
    def loadModel(self, path, name = None):
        if name is None:
            modelName = os.path.join(path,self.name )
        else:
            modelName = os.path.join(path, name)
        try:
            model = tf.keras.models.load_model(modelName)
            print('Loaded model ',modelName)
        except:
            model = -1
            if os.path.isfile(modelName):
                print('faild to load model')
            else:
                print(modelName, '      No such model')
        return model
        
    def discountRewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = np.zeros([rewards.shape[0]])
        for t in reversed(range(0, rewards.shape[0])):
            running_add[rewards[:,t] != 0] = 0
            running_add = running_add * self.gamma + rewards[:,t]
            discounted_rewards[:,t] = running_add
        return discounted_rewards
    
    def train(self):
        y = self.lastMe.reshape((self.lastMe.size), order = 'F')
        pro = self.prob.reshape((self.prob.size//2,2), order = 'F')
        states = self.states.reshape((self.states.shape[1]*self.states.shape[0],self.states.shape[2]), order = 'F')
        rewards = self.lastScore.reshape(self.lastScore.size, order = 'F')

        #print(y, pro)

        gradients = y.astype('float32') - pro.T  
        rewards = self.discountRewards(self.lastScore)
        rewards = rewards.reshape(rewards.size, order = 'F')
        rewards = (rewards- np.mean(rewards)) / np.max([np.std(rewards),1])
        gradients *= rewards.reshape([rewards.size])
        X = states
        Y = pro.T + gradients
        for i in range(len(self.lastMe)):
            self.model.train_on_batch(X[self.numberOfRounds*i:self.numberOfRounds*(i+1),:], Y.T[self.numberOfRounds*i:self.numberOfRounds*(i+1),:])
        #self.clearHistory()
        
    def getWeights(self):
        modelWeights = []
        for layer in self.model.layers:
            layerWeights = []
            for weight in layer.get_weights():
                layerWeights.append(weight)
            modelWeights.append(layerWeights)
        return modelWeights
    
    def getModelStructure(self):
        self.model.summary()
        
    def saveModel(self, path ,name = None):
        if name is None:
            name = self.name
        else:
            name = name + '.h5'
        if os.path.isdir(path) is False:
            os.mkdir(path)
        if os.path.isfile(os.path.join(path, name)):
            os.remove(os.path.join(path, name))
        
        self.model.save(os.path.join(path,name + '.h5'))
    
#------------------------------------------------------------------------------

### Included strategies {TitFtat, TitF2tat, RandomeChoice, AlwaysDefect}

#------------------------Basic strategies------------------------------------

class TitFTatAgent(Basic):
    
    def __init__(self, name):
        super(TitFTatAgent, self).__init__ (name)
        
    def chooseAction(self, me, opponent, t):
    
        if t<=1:
            act = 1
        elif t>1:
            act = opponent[t-1]
        return act
    
class TitF2tatAgent(Basic):
    
    def __init__(self, name):
        super(TitF2tatAgent, self).__init__ (name)
        
    def chooseAction(self, me, opponent, t):
    
        if t<=1:
            act = 1
        elif t>1:
            if opponent[t-1] == 0 and opponent[t-2] == 0:
                act = 0
            else:
                act = 1
        return act

class RandomChoiceAgent(Basic):
    
    def __init__(self, name):
        super(RandomChoiceAgent, self).__init__ (name)
        
    def chooseAction(self, me, opponent, t):
        return int(np.rint(np.random.rand()))
    
class AlwaysDefectAgent(Basic):
    
    def __init__(self, name):
        super(AlwaysDefectAgent, self).__init__ (name)
        
    def chooseAction(self, me, opponent, t):
        return 0
    
#-------------------------------- End Basic strategies-----------------------------

#------------------------------- Neural Agents -------------------------------------
    
class Neural200Agent(BasicNeuron):
    
    def __init__(self, name, actionSpace):
        super(Neural200Agent, self).__init__ (name, actionSpace)
        self.inputSize = 10
        self.actionSpace = actionSpace
        self.nLayers = 3
        
        
        self.nNeurons = [5, 5, self.actionSpace]
        self.model = self.makeModel()
        
        
    def chooseAction(self, me, op, t, index = 0):
        
        if t < self.inputSize:
            state = np.random.rand(self.inputSize)
        else:
            state = np.zeros(self.inputSize)

        state[0:np.min([t,self.inputSize])] = op[np.max([t-self.inputSize,0]):t] 
        state = (state - np.mean(state)) / np.max([np.std(state),1]) 
        X = self.model.predict(np.array([state,]))
        rng = np.random.rand()
        act = np.argmax(X)
        if rng < X[0][act]:
            self.prob[index,t,:] = X[0]
        else:
            act = np.argmin(X)
            self.prob[index,t,:] = X[0]
        self.states[index,t,:] =state
        return act
    
    


class Neural10Agent(BasicNeuron):
    
    def __init__(self, name, actionSpace):
        super(Neural10Agent, self).__init__(name, actionSpace)
        self.inputSize = 3
        self.nLayers = 3
        self.nNeurons = [5, 5, self.actionSpace]
        self.model = self.makeModel()


    def chooseAction(self, me, op, t):
        
        #Play TitFtat first 3 rounds
        if t<self.inputSize:
            if t<=1:
                act = 1
            elif t>1:
                act = op[t-1]
                
        elif t>=self.inputSize:
            X = self.model.predict(np.array([op[t-self.inputSize:t],]))
            act = np.argmax(X)
            self.prob.append(X[0])
            self.states.append(np.array(op[t-self.inputSize:t]))
        return act
    

#--------------------------------------End Neural agents ---------------------------------------

#--------------------------------- Strategies Class of 2019 ------------------------------------


# Student1 ##############################################################
#sss
class Student1_200aAgent(Basic):
    def __init__(self, name):
        super(Student1_200aAgent, self).__init__(name)
        self.noppdefets = 0
        self.cont = 0
        self.attack = False
    
    def chooseAction(self, me, opponent, t):

        if (t == 0): 
            return 1

        if(self.attack==False):
            if(opponent[t-1]==1):
                return 1
            else:
                self.attack = True
                self.noppdefets += 1
                self.cont += 1
                return 0
        else:
            if(self.cont < self.noppdefets):
                self.cont += 1
                return 0

            elif (self.cont == self.noppdefets):
                self.cont += 1
                return 1
            else:
                self.attack = False
                self.cont = 0
                return 1

    def resetState(self):        
        self.noppdefets = 0
        self.cont = 0
        self.attack = False
                
class Student1_200bAgent(Basic):
    def __init__(self, name):
        super(Student1_200bAgent, self).__init__(name)
        self.nDefets = 0
        self.nCoop = 0 
                
    def chooseAction(self, me, opponent, t):
        if (t == 0): 
            return 1

        if (opponent[t-1]==0): 
            self.nDefets += 1
        else: 
            self.nCoop += 1

        if (self.nDefets > self.nCoop):
            return 0

        return 1
                
    def resetState(self):        
        self.nDefets = 0
        self.nCoop = 0
                
class Student1_200cAgent(Basic):
    def __init__(self, name):
        super(Student1_200cAgent, self).__init__(name)
        
    def chooseAction(self, me, opponent, t):
        if (t ==0):
            return 1
        if (me[t-1] == 1 and opponent[t-1] == 1):
            return 1
        elif (me[t-1]==1 and opponent[t-1]==0):
            if (np.random.rand() < 13/15): 
                return 1
            else: 
                return 0
        elif (me[t-1] == 0 and opponent[t-1] == 1):
            if (np.random.rand() < 1/5): 
                return 1
            else: 
                return 0
        else:
            if (np.random.rand() <2/5):
                return 1
            else: 
                return 0
                
class Student1_200mAgent(Basic):
    def __init__(self, name):
        super(Student1_200mAgent, self).__init__(name)
        self.nDefets = 0
        self.nCoop = 0 
                
    def chooseAction(self, me, opponent, t):
        if (t == 0): 
            return 1

        if (opponent[t-1]==0): 
            self.nDefets += 1
        else: 
            self.nCoop += 1

        if (self.nDefets > self.nCoop+1): 
            return 0

        return 1
                
    def resetState(self):        
        self.nDefets = 0
        self.nCoop = 0
                
#End      
                
# Student2 ############################################################################
                
class Student2_200aAgent(Basic):
    def __init__(self,name):
#<<<<<<< Updated upstream
        super(Student2_200aAgent, self).__init__(name)
        #self.me=np.
        self.r=np.zeros(3)
        self.evil=False
        
    def chooseAction(self, me, opponent, t):
        if(t == 0):
            return me[0]
            
         ##Continue..

class Student2_200cAgent(Basic):                
    def __init__(self,name):
        super(Student2_200cAgent, self).__init__(name)
        
        self.maxD = 0
        
    def chooseAction(self, me, opponent, t):    
        return 1
            
# 

class Student13_200aAgent(Basic):
    def __init__(self, name):
        super(Student13_200aAgent, self).__init__(name)
    def chooseAction(self, me, opponent, t):
        if (t == 1):
            return 1

        if (opponent[t-3] == 0 and opponent[t-1] == 0):
            return 1

        if (opponent[t-2] == 0):
            return 0

        return 0


class Student13_200bAgent(Basic):
    def __init__(self, name):
        super(Student13_200bAgent, self).__init__(name)
        self.good = False
                
    def chooseAction(self, me, opponent, t):
        if (np.random.rand() < 0.01):
            self.good = True

        if (self.good):
            return 1

        return 0
       
    def resetState(self):        
        self.good = False


class Student13_200cAgent(Basic):
    def __init__(self, name):
        super(Student13_200cAgent, self).__init__(name)
    def chooseAction(self, me, opponent, t):
        if (t <= 4):
            return 1

        if ((opponent[t-3] == 0 and opponent[t-1] == 0) or (opponent[t-4] == 1 and opponent[t-2] == 1)):
            return 0

        if (opponent[t-5] == 0):
            return 0

        return 0


class Student13_200mAgent(Basic):
    def __init__(self, name):
        super(Student13_200mAgent, self).__init__(name)
    def chooseAction(self, me, opponent, t):
        if (t <= 4):
            return 1

        if (opponent[t-5] == 1 and opponent[t-1] == 1):
            return 1

        return 0


class Student14_200aMSAgent(Basic):
    def __init__(self, name):
        super(Student14_200aAgent, self).__init__(name)
    def chooseAction(self, me, opponent, t):
        patternForMaster = [0, 1, 1, 0, 0, 1, 1]
        patternForSlave = [0, 0, 1, 0, 1, 1, 0]
        patternFound = False
        startingArray = opponent[:8]

        if ( t <= 6 ):
            return patternForSlave[t]

        else:
            for i in range(6):
                if ( startingArray[i] != patternForMaster[i] ):
                    patternFound =  False
                    break
                else:
                    patternFound = True

            if ( patternFound ):
                return 1
            else:
                return opponent[t-1]


class Student14_200bMSAgent(Basic):
    def __init__(self, name):
        super(Student14_200bAgent, self).__init__(name)
        
    def chooseAction(self, me, opponent, t):

        patternForMaster = [0, 1, 1, 0, 0, 1, 0]
        patternForSlave = [0, 0, 1, 0, 1, 1, 1]
        patternFound = False
        startingArray = opponent[:8]

        if ( t <= 6  and t > 0 ): 
            for i in range (t):
                if ( startingArray[i] != patternForSlave[i] ):
                    return opponent[t-1]

        if ( t <= 6 ):
            return patternForMaster[t]

        if ( t > 6 ):
            for i in range(6):
                if ( startingArray[i] != patternForSlave[i] ):
                    patternFound =  False
                    break
                else:
                    patternFound = True

            if ( patternFound ):
                return 0
            else:
                if (t >= me.length -2):
                    return 0
                else:
                    return opponent[t-1]


class Student15_200aMSAgent(Basic):
    def __init__(self, name):
        super(Student15_200aAgent, self).__init__(name)
        self.paired = False
        self.hostile = False
                
    def chooseAction(self, me, opponent, t):
        CODE = [1,0,1,1,0,1,1,1,1,1,0,1]

        if (t == 0):
            return 1

        if (self.paired):
            return 1

        if (self.hostile):
             return 0
        else:
            if (t == 1 and opponent[t-1] == CODE[t-1]):
                return CODE[t]
            elif (t == 1):
                self.hostile = True
    
            if (t == 2 and opponent[t-1] == CODE[t-1]):
                return CODE[t]
            elif (t == 2):
                self.hostile = True

            if (t == 3 and opponent[t-1] == CODE[t-1]):
                return CODE[t]
            elif (t == 3):
                self.hostile = True

            if (t == 4 and opponent[t-1] == CODE[t-1]):
                return CODE[t]
            elif (t == 4):
                self.hostile = True

            if (t == 5 and opponent[t-1] == CODE[t-1]):
                return CODE[t]
            elif (t == 5):
                self.hostile = True

            if (t == 6 and opponent[t-1] == CODE[t-1]):
                return CODE[t]
            elif (t == 6):
                self.hostile = True

            if (t == 7 and opponent[t-1] == CODE[t-1]):
                return CODE[t]
            elif (t == 7):
                self.hostile = True

            if (t == 8 and opponent[t-1] == CODE[t-1]):
                return CODE[t]
            elif (t == 8):
                self.hostile = True

            if (t == 9 and opponent[t-1] == CODE[t-1]):
                return CODE[t]
            elif (t == 9):
                self.hostile = True

            if (t == 10 and opponent[t-1] == CODE[t-1]):
                return CODE[t]
            elif (t == 10):
                self.hostile = True

            if (t == 11 and opponent[t-1] == CODE[t-1]):
                self.paired=True
                return CODE[t]
            elif (t == 11):
                self.hostile = True

        return opponent[t-1]
       
    def resetState(self):        
        self.paired = False
        self.hostile = False





# Student16_200aAgent: Don't know how to implement in python

class Student16_200bAgent(Basic):
    def __init__(self, name):
        super(Student16_200bAgent, self).__init__(name)
        self.numDefects = 0
        self.maxDefects = 100
                
    def chooseAction(self, me, opponent, t):
        if(t==199):
            return 0

        if (t < self.maxDefects):
            return 1

        for i in range(t):
            if (opponent[i] == 0) :
                self.numDefects = self.numDefects + 1

            if (self.numDefects > self.maxDefects):
                return 0
            else:
                return 1
       
    def resetState(self):        
        self.numDefects = 0
        self.maxDefects = 100


class Student16_200cAgent(Basic): 
    def __init__(self, name):
        super(Student16_200cAgent, self).__init__(name)
    def chooseAction(self, me, opponent, t):

        if (t <= 0):
            return 0

        if (t==1):
            if (opponent[0] == 0):
                return 0
            else:
                return 1

        elif (t==2):
            if (opponent[0] == 0 or opponent[1] == 0):
                return 0;
            else:
                return 1

        else:
            if (opponent[t-1] == 0 or opponent[t-2]==0 or opponent[t-3]==0):
                return 0
            else:
                return 1

        return 0


# Student16_200mAgent: Same as Student16_200aAgent


class Student17_200aAgent(Basic): 
    def __init__(self, name):
        super(Student17_200aAgent, self).__init__(name)
    def chooseAction(self, me, opponent, t):

        maxDefects = 2
        numDefects = 0

        if (t < maxDefects):
            return 1

        for i in range(t):
            if (opponent[i] == 0):
                numDefects = numDefects + 1
            if (numDefects > maxDefects):
                return np.random.randint(0,2)
            else:
                return 0


class Student17_200bAgent(Basic): 
    def __init__(self, name):
        super(Student17_200bAgent, self).__init__(name)
    def chooseAction(self, me, opponent, t):

        maxDefects = 5
        numDefects = 0

        if (t < maxDefects):
            return 1

        for i in range(t):
            if (opponent[i] == 0):
                numDefects = numDefects + 1
            if (numDefects > maxDefects):
                return np.random.randint(0,2)
            else:
                return 0

# Student17_200cAgent = Student17_200bAgent


class Student17_200mAgent(Basic):
    def __init__(self, name):
        super(Student17_200mAgent, self).__init__(name)
        self.angel = False
                
    def chooseAction(self, me, opponent, t):

        if (np.random.rand() < 0.1):
            self.angel = False

        if (self.angel):
            return 1

        return 0
       
    def resetState(self):        
        self.angel = False


# Student18_200x aren't interesting in this case


class Student19_200aAgent(Basic):
    def __init__(self, name):
        super(Student19_200aAgent, self).__init__(name)
    def chooseAction(self, me, opponent, t):

        C = 0
        D = 0

        if (t == 0):
            return 0

        if (opponent[t-1] == 1):
            C = C + 1
        else :
            D = D + 1

        if (C > D):
            return 1

        if (C < D):
            return 0
        else:
            return 0 


class Student19_200bAgent(Basic):   
    def __init__(self, name):
        super(Student19_200bAgent, self).__init__(name)
    def chooseAction(self, me, opponent, t):

        C = 0
        D = 0

        if (t == 0):
            return 1

        if (opponent[t-1] == 1):
            C = C + 1
        else:
            D = D + 1

        if (C > D):
            return 0

        if (C < D):
            return 1
        else:
            return 0


class Student19_200cAgent(Basic):  
    def __init__(self, name):
        super(Student19_200cAgent, self).__init__(name)
    def chooseAction(self, me, opponent, t):

        C = 0
        D = 0

        if (t < 5):
            return 0

        if (opponent[t-1] == 1):
            C = C + 1
        else:
            D = D + 1

        if (C/(C+D) > 0.75):
            return 1
        else:
            return 0

class Student19_200mAgent(Basic): 
    def __init__(self, name):
        super(Student19_200mAgent, self).__init__(name)
    def chooseAction(self, me, opponent, t):

        C = 0
        D = 0
        if (t < 5):
            return 0
        else:
            sum_ = 0
            avg = 0
            for i in range(t-5,t-2):
                sum_ += opponent[i]
                avg = sum_/5

            if (avg >= 0.6):
                return 1
            else:
                return 0


# Student20_200x aren't interesting in this case


class Student21_200aAgent(Basic):
    def __init__(self, name):
        super(Student21_200aAgent, self).__init__(name)
        self.against_self_a = False
        self.against_self_b = False
        self.against_Tf2T = False
                
    def chooseAction(self, me, opponent, t):
        if (t == 0):
            return 0

        if (opponent[0] == 0):
            if (t == 1):
                return 0

            if (t == 2 and opponent[t-1] == 0):
                self.against_self_a = True
                return 1

            if (self.against_self_a == True):
                if (opponent[t-1] == 0):
                    self.against_self_a = False
                    return 0
                return 1

            if (t == 2 and opponent[t-1] == 1):
                self.against_self_b = True
                return 0

            if (self.against_self_b == True):
                if (opponent[t-1] == 0):
                    self.gainst_self_b = False
                    return 0
                return 0

        if (t == 1):
            return 1

        if (t == 2):
            if (opponent[t-1] == 1):
                self.against_Tf2T = True
                return 0
            return 1

        if (self.against_Tf2T == True):
            if (opponent[t-1] == 0 and (me[t-1] == 1 or me[t-2] == 1)):
                self.against_Tf2T = False
                return 0
            if me[t-1] == 1:
                return 0
            else:
                return 1

        if (opponent[t-1] == 0):
            return 0

        return 1
       
    def resetState(self):        
        self.against_self_a = 0
        self.against_self_b = 0
        self.against_Tf2T = 0


# Student21_200bAgent = Student21_200cAgent
class Student21_200cAgent(Basic):
    def __init__(self, name):
        super(Student21_200cAgent, self).__init__(name)
        self.against_self_a = 0
                
    def chooseAction(self, me, opponent, t):

        if (t == 0):
            return 0

        if (opponent[0] == 0):
            if (t == 1):
                return 1
            if (t == 2 and opponent[t-1] == 0):
                self.against_self_a = True
                return 1
            if (self.against_self_a == True):
                if (opponent[t-1] == 1):
                    self.against_self_a = False
                    return 0
                return 1

        return 0
       
    def resetState(self):        
        self.against_self_a = 0


# Student22_200cAgent: Tit for tat
# Student22_200cAgent: For 200 rounds
# Student22_200cAgent: Always defect

class Student22_200mAgent(Basic):
    def __init__(self, name):
        super(Student22_200mAgent, self).__init__(name)
        self.timesToForgive = 4
                
    def chooseAction(self, me, opponent, t):
        if (t == 0):
            return 1

        if (me[t-1] == 0 and me[t-2] == 0 and opponent[t-1] == 0 and opponent[t-2] == 0 and self.timesToForgive > 0):
            self.timesToForgive -= 1
            return 1
        else:
            return opponent[t-1];
       
    def resetState(self):        
        self.timesToForgive = 4


class Student23_200aMSAgent(Basic):
    def __init__(self, name):
        super(Student23_200aAgent, self).__init__(name)
        self.masterFound = True
        self.lackeyFound = True
        self.arrayLength = 8
        self.masterPassword = [0, 0, 1, 1, 0, 1, 1, 1]
        self.lackeyPassword = [0, 1, 0, 1, 1, 0, 0, 1]
                
    def chooseAction(self, me, opponent, t):
        if (t < self.arrayLength):
            return self.lackeyPassword[t]

        if (t == self.arrayLength):
            for i in range(self.arrayLength):
                if (opponent[i] != self.masterPassword[i]):
                    self.masterFound = False
                if (opponent[i] != self.lackeyPassword[i]):
                    self.lackeyFound = False;

        if( self.masterFound or self.lackeyFound):
            return 1
        else:
            return 0
              
    def resetState(self):        
        self.masterFound = True
        self.lackeyFound = True
        self.arrayLength = 8
        self.masterPassword = [0, 0, 1, 1, 0, 1, 1, 1]
        self.lackeyPassword = [0, 1, 0, 1, 1, 0, 0, 1]


class Student23_200bMSAgent(Basic):
    def __init__(self, name):
        super(Student23_200bAgent, self).__init__(name)
        self.lackeyFound = True
        self.arrayLength = 8
        self.masterPassword = [0, 0, 1, 1, 0, 1, 1, 1]
        self.lackeyPassword = [0, 1, 0, 1, 1, 0, 0, 1]
                
    def chooseAction(self, me, opponent, t):
        if (t < arrayLength):
            return self.masterPassword[t]

        if (t == arrayLength ):
            for i in range(self.arrayLength):
                if (opponent[i] != self.lackeyPassword[i]):
                    lackeyFound = False

        if(self.lackeyFound):
            return 0

        else:
            return opponent[t-1]
                
    def resetState(self):        
        self.lackeyFound = True
        self.arrayLength = 8
        self.masterPassword = [0, 0, 1, 1, 0, 1, 1, 1]
        self.lackeyPassword = [0, 1, 0, 1, 1, 0, 0, 1]


class Student23_200cMSAgent(Basic):
    def __init__(self, name):
        super(Student23_200cAgent, self).__init__(name)
        self.masterFound = True
        self.lackeyFound = True
        self.arrayLength = 8
        self.masterPassword = [0, 0, 1, 1, 0, 1, 1, 1]
        self.lackeyPassword = [0, 1, 0, 1, 1, 0, 0, 1]
                
    def chooseAction(self, me, opponent, t):
        if (t < self.arrayLength):
            return self.lackeyPassword[t]

        if (t == self.arrayLength):
            for i in range(self.arrayLength):
                if (opponent[i] != self.masterPassword[i]):
                    self.masterFound = False
                elif (opponent[i] != self.lackeyPassword[i]):
                    self.lackeyFound = False

        if( (t > self.arrayLength + 5) and (opponent[t-1] == 0) ):
            self.lackeyFound = False

        if( self.masterFound or self.lackeyFound):
            return 1
        else:
            return 0
                
    def resetState(self):        
        self.masterFound = True
        self.lackeyFound = True
        self.arrayLength = 8
        self.masterPassword = [0, 0, 1, 1, 0, 1, 1, 1]
        self.lackeyPassword = [0, 1, 0, 1, 1, 0, 0, 1]


class Student23_200mAgent(Basic):
    def __init__(self, name):
        super(Student23_200mAgent, self).__init__(name)
        self.trust = True
        self.brokenTrust = False
        self.numBreach = 0
        self.maxBreach = 25
                
    def chooseAction(self, me, opponent, t):
        if (t < 1):
            return 1

        if(self.numBreach > self.maxBreach):
            brokenTrust = True

        if (self.brokenTrust):
            return 0

        if ( self.trust ):
            if( opponent[t-1] != 1 ):
                self.trust = False
                self.numBreach+=1
        else:
            if ( me[t-1] == 1 and opponent[t-1] == 1 ):
                self.trust = True

        if ( t >= 3 and opponent[t-1] == 0 and opponent[t-2] == 0 and opponent[t-3] == 0
            and me[t-1] == 1 and me[t-2] == 1 and me[t-3] == 1):
            return 0;

        return 1
                
    def resetState(self):        
        self.trust = True
        self.brokenTrust = false
        self.numBreach = 0
        self.maxBreach = 25


# Student24_200aAgent is the same as Student26_200cAgent

class Student24_200bAgent(Basic):
    def __init__(self, name):
        super(Student24_200bAgent, self).__init__(name)
        self.coops = 0
        self.betrayals = 0
        self.r = 0
        self.threshold = 0
                
    def chooseAction(self, me, opponent, t):
        self.r = np.random.rand()
        self.threshold = self.coops/(self.coops+self.betrayals + 1e-7)

        if (self.r < self.threshold):
            return 1
        else:
            return 0
                
    def resetState(self):        
        self.coops = 0
        self.betrayals = 0
        self.r = 0
        self.threshold = 0


class Student24_200cAgent(Basic):
    def __init__(self, name):
        super(Student24_200cAgent, self).__init__(name)
        self.betrayals = 0
        self.rounds = 0
        self.threshold = 0.25
        self.ratio = 0
                
    def chooseAction(self, me, opponent, t):
        self.rounds = self.rounds + 1

        if (opponent[t-1] == 0):
            self.betrayals = self.betrayals + 1

        self.ratio = self.betrayals / self.rounds;

        if (self.ratio < self.threshold):
            return 1
        else :
            return 0
                
    def resetState(self):        
        self.betrayals = 0
        self.rounds = 0
        self.threshold = 0.25
        self.ratio = 0


class Student24_200mAgent(Basic):
    def __init__(self, name):
        super(Student24_200mAgent, self).__init__(name)
        self.isResponding = False
        self.retaliations = 0
        self.betrayals = 0
        self.appeasements = 0
                
    def chooseAction(self, me, opponent, t):
        if (t <= 0):
            return 1

        if (self.isResponding == False):
            if (opponent[t-1] == 0):
                self.betrayals = self.betrayals + 1
                self.retaliations = self.betrayals
                self.appeasements = 2
                self.isResponding = True

        if (self.isResponding == True):
            if (self.retaliations > 0):
                self.retaliations = self.retaliations - 1
                return 0
        elif (self.appeasements > 0):
            self.appeasements = self.appeasements - 1
            return 1
        else:
            self.isResponding = False

        return 1; 
                
    def resetState(self):        
        self.isResponding = False
        self.retaliations = 0
        self.betrayals = 0
        self.appeasements = 0


# Student26_200aAgent: Tit for tat
class Student26_200bAgent(Basic): 
    def __init__(self, name):
        super(Student26_200bAgent, self).__init__(name)
    def chooseAction(self, me, opponent, t):
        if (t == 0):
            return 1

        numDefects = 0
        for i in range(t):
            if (opponent[i] == 0):
                numDefects = numDefects + 1

        numCoop = 0
        if(opponent[t-1]==1):
            return 1

        numDefectsMe = 0
        for i in range(t,0,-1):
            if (me[i-1] == 0):
                numDefectsMe = numDefectsMe + 1
            if (me[i-1]==1):
                break

        if (numDefects > numDefectsMe):
            return 0
        else:
            return 1; # Otherwise cooperate

        return 1;


class Student26_200cAgent(Basic):  
    def __init__(self, name):
        super(Student26_200cAgent, self).__init__(name)
    def chooseAction(self, me, opponent, t):

        if (t == 0):
            return 1

        numDefects = 0
        for i in range(t):
            if (opponent[i] == 0): 
                numDefects = numDefects + 1

        numCoop = 0
        if(opponent[t-1]==1 and opponent[t-2]==1):
            return 1

        numDefectsMe = 0
        for i in range(t,0,-1):
            if (me[i-1] == 0):
                numDefectsMe = numDefectsMe + 1
            if (me[i-1]==1):
                break

        if (numDefects > numDefectsMe):
            return 0
        else:
            return 1

        return 1


class Student26_200mAgent(Basic): 
    def __init__(self, name):
        super(Student26_200mAgent, self).__init__(name)
    def chooseAction(self, me, opponent, t):

        if (t == 0):
            return 1

        if (opponent[t-2]!=me[t-1]):
            return 1

        return opponent[t-1]
