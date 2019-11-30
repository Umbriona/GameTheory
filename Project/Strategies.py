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