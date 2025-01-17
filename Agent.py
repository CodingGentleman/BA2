import torch as T
import numpy as np
from neural_network import DeepQNetwork

class Agent():
    # gamma: discount factor of future rewards
    # epsilon: parameter if the agent should learn
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=100000, eps_end=0.01, eps_dec=2e-5):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.mem_size = max_mem_size
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.mem_cntr = 0

        self.action_space = [i for i in range(n_actions)]
        self.Q = DeepQNetwork(lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)
        
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def train(self):
        self.Q.train()

    def eval(self):
        self.Q.eval()

    def save_checkpoint(self, target_file="checkpoint.pth.tar"):
        print('--- saving checkpoint ---')
        checkpoint = {'state_dict' : self.Q.state_dict(), 'optimizer' : self.Q.optimizer.state_dict()}
        T.save(checkpoint, target_file)

    def load_checkpoint(self, source_file="checkpoint.pth.tar"):
        print('--- loading checkpoint ---')
        checkpoint = T.load(source_file)
        self.Q.load_state_dict(checkpoint['state_dict'])
        self.Q.optimizer.load_state_dict(checkpoint['optimizer'])

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal
        self.mem_cntr += 1

    def choose_action(self, observation, evaluation=False):
        if evaluation or np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q.device)
            actions = self.Q.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        # do not learn until the memory has at least the learning batch size
        if self.mem_cntr < self.batch_size:
            return
        
        # clear up before learning - because we do batch learning
        self.Q.optimizer.zero_grad()

        # position of the maximum memory to select a random batch up to this point
        max_mem = min(self.mem_cntr, self.mem_size)

        # select the batch to learn from. replace False so we don't select the same memory more than once
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # convert the numpy array subset of the memory to a pytorch tensor
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q.device)

        # same convertion for the new state, reward and terminal
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q.device)

        action_batch = self.action_memory[batch]

        # feed forward to the neural network to get the relevant parameters for the loss function
        # estimate of the current state
        q_eval = self.Q.forward(state_batch)[batch_index, action_batch]
        # estimae of the future state
        q_next = self.Q.forward(new_state_batch)

        # expected reward on terminal is 0 by definition
        q_next[terminal_batch] = 0.0

        # getting the maximum value of the future reward times gamma (future rewards are accounted a little bit less) 
        q_target = reward_batch + self.gamma * T.max(q_next,dim=1)[0]

        loss = self.Q.loss(q_target, q_eval).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()

        # calculate epsilon according to decay
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min