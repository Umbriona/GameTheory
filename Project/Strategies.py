import numpy as np
import tensorflow as tf
import functools
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import softmax
from tensorflow.keras.optimizers import Adam

#----------------------------------Parent Classes------------------------------
class Basic():
    
    def __init__(self,name):
        self.name = name
        self.lastScore = []
        self.lastMe = []
        self.lastOp = []
        
    def chooseAction(self):
        return 1
    
    def clearHistory(self):
        self.lastScore = []
        self.lastMe = []
        self.lastOp = []

class BasicNeuron(Basic):

    def __init__(self,name, actionSpace):
        super(BasicNeuron, self).__init__(name)
        self.actionSpace = actionSpace
        self.learning_rate = 0.001
        self.gamma = 0.9
        self.prob = []
        self.states = []
        
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
    
    def clearHistory(self):
        self.lastScore = []
        self.lastMe = []
        self.lastOp = []
        self.states = []
        self.prob = []
        
    def discountRewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = np.zeros([rewards.shape[0]])
        for t in reversed(range(0, rewards.shape[1])):
            running_add[rewards[:,t] != 0] = 0
            running_add = running_add * self.gamma + rewards[:,t]
            discounted_rewards[:,t] = running_add
        return discounted_rewards
    
    def train(self):
        y = np.zeros([2, (self.lastMe[0].size - 10) * len(self.lastMe)])
        count = 0
        for i in self.lastMe:

            for j in range(np.size(i) - 10):
                y[int(i[j]),count*(i.size-10) + j] = 1
            count += 1
        pro = np.asarray(self.prob)  / (np.sum(np.asarray(self.prob), axis = 1)[0] + 1e-7)
        gradients = np.array(y).astype('float32') - pro.T         
        rewards = np.asarray(self.lastScore)[:,10:]
        rewards = self.discountRewards(rewards)

        rewards = rewards / (np.std(rewards - np.mean(rewards)) + 1e-7)
        gradients *= rewards.reshape([rewards.size])
        X = np.asarray(self.states)
        Y = np.asarray(self.prob).T + self.learning_rate * gradients
        for i in range(len(self.lastMe)):
            self.model.train_on_batch(X[190*i:190*(i+1),:], Y.T[190*i:190*(i+1),:])
        self.clearHistory()
        
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
        
    def saveModel(self):
        
        self.model.save('')
    
#------------------------------------------------------------------------------

### Included strategies {TitFtat, TitF2tat, RandomeChoice, AlwaysDefect}

#------------------------Basic strategies------------------------------------

class TitFtatAgent(Basic):
    
    def __init__(self, name):
        super(TitFtatAgent, self).__init__ (name)
        
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
        
        
    def chooseAction(self, me, op, t):
        
        #Play TitFtat first 10 rounds
        if t<self.inputSize:
            if t<=1:
                act = 1
            elif t>1:
                act = op[t-1]
                
        elif t>=self.inputSize:
            X = self.model.predict(np.array([op[t-10:t],]))
            rng = np.random.rand()
            act = np.argmax(X)
            if rng < X[0][act]:
                self.prob.append(X[0])
            else:
                act = np.argmin(X)
                self.prob.append(X[0])
            self.states.append(np.array(op[t-10:t]))
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
        super(atienza200aAgent, self).__init__(name)
        self.noppdefets = 0
        self.cont = 0
        self.attack = False
    
    def chooseAction(me, opponent, t):

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

    def resetState():        
        self.noppdefets = 0
        self.cont = 0
        self.attack = False
                
class Student1_200bAgent(Basic):
    def __init__(self, name):
        super(atienza200bAgent, self).__init__(name)
        self.nDefets = 0
        self.nCoop = 0 
                
    def chooseAction(me, opponent, t):
        if (t == 0): 
            return 1

        if (opponent[t-1]==0): 
            self.nDefets += 1
        else: 
            nCoop += 1

        if (self.nDefets > self.nCoop): 
            return 0

        return 1
                
    def resetState():        
        self.nDefets = 0
        self.nCoop = 0
                
class Student1_200cAgent(Basic):
    def __init__(self, name):
        super(atienza200_cAgent, self).__init__(name)
        
    def chooseAction(me, opponent, t):
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
            if (np.random.rand < 1/5): 
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
                
    def chooseAction(me, opponent, t):
        if (t == 0): 
            return 1

        if (opponent[t-1]==0): 
            self.nDefets += 1
        else: 
            nCoop += 1

        if (self.nDefets > self.nCoop+1): 
            return 0

        return 1
                
    def resetState():        
        self.nDefets = 0
        self.nCoop = 0
                
#End      
                
# Student2 ############################################################################
                
class Student2_200aAgent(Basic):
    def __init__(self,name):
        super(Student2_200aAgent, self).__init__(name)
        
    def chooseAction(me, opponent, t):
        me=3*[0]
        me[1]

        evil=False
        if(t == 0):
            return me[0]
        if (t ==1):
            if(opponent[t-1] == 0):
                me[1]=round(np.random.rand)
                return me[1]
        
        if (t == 2):
            if(opponent[t-1] == me[0]):
                me[2]=1
                return 1
        
        if (evil == True):
            me[0]=0
            return 0
        
        if (opponent[t-1] == 0 and opponent[t-2] == 0 and opponent[t-3] == 0):
            evil=True
            return 0
        elif (opponent[t-1] == me[1]):
            return 1
        else:
            return 0
            

class Student2_200cAgent(Basic):                
    def __init__(self,name):
       super(Student2_200cAgent, self).__init__(name)
        
    def chooseAction(me, opponent, t):    
            
        maxDefects = 5
        if (t < maxDefects):
            return 1

        numDefects = t - sum(opponent.slice(0,t))

        if numDefects >= maxDefects:
            return 0

        return 1

        
## NOT FINISHED
                
# Student3 ############################################################################            

class Student3_200aAgent(Basic):
    def __init__(self,name):
        super(Student3_200aAgent, self).__init__(name)
        
    def chooseAction(me, opponent, t):  
    
        if (t < 6):
            return 1
    
        if (opponent[t-6] == 0 and np.random.rand < 0.2):
            return 0
    
        if (opponent[t-5] == 0 and np.random.rand < 0.2):
            return 0
    
        if (opponent[t-4] == 0 and np.random.rand < 0.2):
            return 0    
        
        if (opponent[t-3] == 0 and np.random.rand < 0.2):
            return 0
    
        if (opponent[t-2] == 0 and np.random.rand < 0.2):
            return 0 
    
        if (opponent[t-1] == 0 and np.random.rand < 0.2):
            return 0

    
class Student3_200bAgent(Basic):
    def __init__(self,name):
        super(Student3_200bAgent, self).__init__(name)    
    
    def chooseAction(me, opponent, t): 
        if (t < 3):
            return 1
        
        if (t > 180):
            if (np.random.rand < 0.05):
                return 0
    
        if (opponent[t-3] + opponent[t-2] + opponent[t-1] < 1.5):
            return 0
        
        return 1
    
    
# Student4 ############################################################################        
        
class Student4_200aAgent(Basic):
    def __init__(self,name):
        super(Student4_200aAgent, self).__init__(name)    

        
    def chooseAction(me, opponent, t): 
        T=0
        p=0
        i=0
        
        if (t > 9):
            for i in range(t):
                T=T+opponent[i]
        
        p=T/(t+1)
        
        if (np.random.rand > p):
            return 0
        
        return 1

# Student5 ############################################################################   
        
class Student5_200aAgent(Basic):
    def __init__(self,name):
        super(Student5_200aAgent, self).__init__(name)  
        
    def chooseAction(me, opponent, t):
        if (t <= 1):
            return 1
        
        if ( (me[t-2] == 0 and me[t-1] == 0) or (me[t-2] == 1 and me[t-1] == 1) ):
            return 0
        else: 
            return 1
        
        
class Student5_200bAgent(Basic):
    def __init__(self,name):
        super(Student5_200bAgent, self).__init__(name)  
        
    def chooseAction(me, opponent, t):     
        evil=False
        if (np.random.rand < 0.05):
            evil=True 
        
        if (np.random.rand <- 0.02):
            evil=False
        
        if (evil==True):
            return 0
        
        return 1
    
class Student5_200cAgent(Basic):
    def __init__(self,name):
        super(Student5_200cAgent, self).__init__(name)  

    def chooseAction(me, opponent, t):
        maxDefects = 0
        numDefects = 0
        i = 0
        if (t < maxDefects):
            return 1
        
        for i in range(t):
            if (opponent[i] == 0):
                numDefects = numDefects + 1
        
        if (numDefects > maxDefects):
            return 0
        else:
            return 1

class Student5_200mAgent(Basic):
    def __init__(self,name):
        super(Student5_200cAgent, self).__init__(name)  

    
    def chooseAction(me, opponent, t): 
        numDefects = 0
        numCooperate = 0
        i = 0
        for i in range(t+1):
            if (me[i] == 0):
                numDefects = numDefects + 1
            else:
                numCooperate = numCooperate + 1
        
        if (numCooperate == numDefects):
            return round(np.random.rand)
        
        elif (numDefects > numCooperate):
            return 1
        
        else:
            return 0

#Student 6 not implemented
        
# Student7 ############################################################################   
            
class Student7_200aAgent(Basic):
    def __init__(self,name):
        super(Student7_200aAgent, self).__init__(name) 
    
    def chooseAction(me, opponent, t):
        trust=True
        if (t >= 3 and trust):
            actionSum=functools.reduce(lambda a,b : a+b,opponent)
            trust = (actionSum/(t+1)) > 0.6
            
        if (trust):
            if(t==0):
                return 1
            return opponent[t-1]
        return 0        

class Student7_200bAgent(Basic):
    def __init__(self,name):
        super(Student7_200aAgent, self).__init__(name) 
            
    def chooseAction(me, opponent, t):
          tft = False
          tf2t = False
          itft = False
          alwaysDefect = False
          alwaysCooperate = False

          startActions = [1, 0, 0, 0, 1, 1]
          
          if (t < len(startActions)):
              return startActions[t]
          elif (t == len(startActions)):
              oppResponse=slice(opponent[0],opponent[t],1)
                  
          if (oppResponse[1] == 1 and oppResponse[2] == 0 and oppResponse[3] == 0 and oppResponse[5] == 1): 
              tft = True
          elif (oppResponse[2] == 1 and oppResponse[3] == 0 and oppResponse[4] == 0 and oppResponse[5] == 1):
              tf2t = True
          elif (oppResponse[1] == 0 and oppResponse[2] == 1 and oppResponse[3] == 1 and oppResponse[5] == 0):
              itft = True;
          elif (oppResponse[1] == 0 and oppResponse[2] == 0 and oppResponse[3] == 0 and oppResponse[5] == 0):
              alwaysDefect = True;
          elif (oppResponse[1] == 1 and oppResponse[2] == 1 and oppResponse[3] == 1 and oppResponse[4] == 1 and oppResponse[5] == 1):
              alwaysCooperate = True;
      
          if (t >= len(startActions)):
  
              if (tft):
                  return opponent[t-1]
              elif (tf2t):
                  return abs(me[t-1] - 1)
              elif (itft or alwaysDefect or alwaysCooperate):
                  return 0;
              else:
                  return opponent[t-1]
      
class Student7_200mAgent(Basic):
    def __init__(self,name):
        super(Student7_200mAgent, self).__init__(name)           
          
    def chooseAction(me, opponent, t):   
        trust = True
        reset = False
        
        if (t >=3 and trust):
            oppActionSum=functools.reduce(lambda a,b : a+b,opponent)
            trust=(oppActionSum/(t+1)) > 0.6
        elif (t >= 4 and trust==False):
            oppLastMoves = slice(opponent[t-4],opponent[t])
            trust = (oppLastMoves[0] == 1 and oppLastMoves[1] == 1 and oppLastMoves[2] == 1 and oppLastMoves[3] == 1)
        
        if(trust):
            if(t == 0):
                return 1
            if(reset):
                reset=False
                return 0
            else:
                reset = me[t-1] != opponent[t-1] and me[t-2] != opponent[t-2]
            if(reset):
                return 0
            return 1
        return 0
    
# Student8, probability based, been done before ############################################################################   

# Student9 Hard coded for 200 rounds ############################################################################ 

# Student10 ############################################################################ 
class Student10_200mAgent(Basic):
    def __init__(self,name):
        super(Student10_200mAgent, self).__init__(name)  
        
    def chooseAction(me, opponent, t):
        if (t == 0):
            return 1
        
        if (t <= 9):
            return opponent[t-1]
        
        coop=0
        for i in range(t):
            coop +=opponent[t-i]
        
        def1=10-coop
        coopPercent=coop/10
        if (np.random.rand <= coopPercent):
            return 1
        
        meCoop=0
        for j in range(10):
            meCoop=meCoop+me[t-1]
        
        meDef=10-meCoop
        totDef=meDef+def1
        defPercent=def1/totDef
        if (np.random.rand <= defPercent):
            return 0
        else:
            return 1
        
# Student11 ############################################################################         
class Student11_200mAgent(Basic):
    def __init__(self,name):
        super(Student11_200mAgent, self).__init__(name)              

    def chooseAction(me, opponent, t):
        if (opponent[t-3] == 1):
            if (opponent[t-2] == 1):
                if (opponent[t-1] == 1):
                    if (opponent[t] == 1):
                        return 1
                    return 1
        
                if (opponent[t] == 1):
                    return 1
                return 0
    
            if (opponent[t-1] == 1):
                if (opponent[t] == 1):
                    return 1
                return 0
    
            if (opponent[t] == 1):
                return 0

            return 1

  
        if (opponent[t-2] == 1):
            if (opponent[t-1] == 1):
                if (opponent[t] == 1):
                    return 1
                return 1
    
            if (opponent[t] == 1):
                return 0;

           return 0;
  
        if (opponent[t-1] == 1):
            if (opponent[t] == 1):
                return 1;
            return 1;
  
        if (opponent[t] == 1):
            return 1
        return 1


class Student13_200cAgent(Basic):
    def chooseAction(me, opponent, t):
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
        if (random.random() < 0.01):
            good = True

        if (good):
            return 1

        return 0
       
    def resetState():        
        self.good = False


class Student13_200cAgent(Basic):
    def chooseAction(me, opponent, t):
        if (t <= 4):
            return 1

        if ((opponent[t-3] == 0 and opponent[t-1] == 0) or (opponent[t-4] == 1 and opponent[t-2] == 1)):
            return 0

        if (opponent[t-5] == 0):
            return 0

        return 0


class Student13_200mAgent(Basic):
    def chooseAction(me, opponent, t):
        if (t <= 4):
            return 1

        if (opponent[t-5] == 1 and opponent[t-1] == 1):
            return 1

        return 0


class Student14_200aAgent(Basic):
    def chooseAction(me, opponent, t):
        patternForMaster = [0, 1, 1, 0, 0, 1, 1]
        patternForSlave = [0, 0, 1, 0, 1, 1, 0]
        patternFound = False
        startingArray = opponent.slice(0, 7)

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


class Student14_200bAgent(Basic):
    def chooseAction(me, opponent, t):

        patternForMaster = [0, 1, 1, 0, 0, 1, 0]
        patternForSlave = [0, 0, 1, 0, 1, 1, 1]
        patternFound = False
        startingArray = opponent.slice(0, 7)

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


# Student15_200cAgent = Student15_200aAgent
# Student15_200mAgent: Don't know how to implement in python


class Student15_200aAgent(Basic):
    def __init__(self, name):
        super(Student15_200cAgent, self).__init__(name)
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
       
    def resetState():        
        self.paired = False
        self.hostile = False


# Student15_200bAgent: Same as Student15_200aAgent
# Student15_200cAgent: Same as Student15_200aAgent
# Student15_200mAgent: tit for two tats


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
       
    def resetState():        
        self.numDefects = 0
        self.maxDefects = 100


class Student16_200cAgent(Basic):            
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
    def chooseAction(me, opponent, t):

        maxDefects = 2
        numDefects = 0

        if (t < maxDefects):
            return 1

        for i in range(t):
            if (opponent[i] == 0):
                numDefects = numDefects + 1
            if (numDefects > maxDefects):
                return round(random.random())
            else:
                return 0


class Student17_200bAgent(Basic):               
    def chooseAction(me, opponent, t):

        maxDefects = 5
        numDefects = 0

        if (t < maxDefects):
            return 1

        for i in range(t):
            if (opponent[i] == 0):
                numDefects = numDefects + 1
            if (numDefects > maxDefects):
                return round(random.random())
            else:
                return 0

# Student17_200cAgent = Student17_200bAgent


class Student17_200mAgent(Basic):
    def __init__(self, name):
        super(Student17_200mAgent, self).__init__(name)
        self.angel = False
                
    def chooseAction(self, me, opponent, t):

        if (random.random() < 0.1):
            self.angel = False

        if (self.angel):
            return 1

        return 0
       
    def resetState():        
        self.angel = False


# Student18_200x aren't interesting in this case


class Student19_200aAgent(Basic):                
    def chooseAction(me, opponent, t):

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
    def chooseAction(me, opponent, t):

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
    def chooseAction(me, opponent, t):

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
    def chooseAction(me, opponent, t):

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

            if (against_self_a == True):
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
       
    def resetState():        
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
       
    def resetState():        
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
       
    def resetState():        
        self.timesToForgive = 4


class Student23_200aAgent(Basic):
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
              
    def resetState():        
        self.masterFound = True
        self.lackeyFound = True
        self.arrayLength = 8
        self.masterPassword = [0, 0, 1, 1, 0, 1, 1, 1]
        self.lackeyPassword = [0, 1, 0, 1, 1, 0, 0, 1]


class Student23_200bAgent(Basic):
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
                
    def resetState():        
        self.lackeyFound = True
        self.arrayLength = 8
        self.masterPassword = [0, 0, 1, 1, 0, 1, 1, 1]
        self.lackeyPassword = [0, 1, 0, 1, 1, 0, 0, 1]


class Student23_200cAgent(Basic):
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
                
    def resetState():        
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
                
    def resetState():        
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
        self.r = random.random()
        self.threshold = self.coops/(self.coops+self.betrayals)

        if (self.r < self.threshold):
            return 1
        else:
            return 0
                
    def resetState():        
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
                
    def resetState():        
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
                
    def resetState():        
        self.isResponding = False
        self.retaliations = 0
        self.betrayals = 0
        self.appeasements = 0


# Student26_200aAgent: Tit for tat
class Student26_200bAgent(Basic):                
    def chooseAction(me, opponent, t):
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
    def chooseAction(me, opponent, t):

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
    def chooseAction(me, opponent, t):

        if (t == 0):
            return 1

        if (opponent[t-2]!=me[t-1]):
            return 1

        return opponent[t-1]
