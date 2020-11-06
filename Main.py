import os.path
import numpy as np
from classic_algorithms import FirstFit
from classic_algorithms import NextFit
from classic_algorithms import BestFit
from environment import BinEnvironment
from environment import ItemProvider
from agent import Agent
from utils import plot_learning
from utils import read_csv

max_simultaneously_bins = 5
load_checkpoint = True
filename = "flat"
data = read_csv(file="training_data/%s.csv" % filename)
print(data)

agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01, input_dims=[2], lr=0.001)
agent.train()
if load_checkpoint and os.path.isfile("%s_checkpoint.pth.tar" % filename):
    agent.load_checkpoint(source_file="%s_checkpoint.pth.tar" % filename)

item_provider = ItemProvider(sample_size=100, data=data, randomize=True)
env = BinEnvironment(max_simultaneously_bins, item_provider=item_provider)

scores, eps_history = [], []
n_games = 100

for i in range(n_games):
    score = 0
    done = False
    if i % 5 == 0:
        agent.save_checkpoint(target_file="%s_checkpoint.pth.tar" % filename)
    if i == n_games-1:
        data = np.random.choice(data, size=100, replace=False)
        print(data)
        item_provider = ItemProvider(sample_size=100, data=data, randomize=True)
        env = BinEnvironment(max_simultaneously_bins, item_provider=item_provider)
        agent.eval()
    observation = env.reset()
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        score += reward
        agent.store_transition(observation, action, reward, observation_, done)
        agent.learn()
        observation = observation_
    scores.append(score)
    eps_history.append(agent.epsilon)
    avg_score = np.mean(scores[-100:])
    print('epoch ', i, 'score %.2f' % score, 'average score %.2f' % avg_score, 'epsilon %.2f' % agent.epsilon, 'bin count %i' % env.bin_count)


next_fit = NextFit(max_simultaneously_bins)
first_fit = FirstFit(max_simultaneously_bins)
best_fit = BestFit(max_simultaneously_bins)
item_provider.reset()
while item_provider.has_next():
    item = item_provider.next()
    next_fit.put(item)
    first_fit.put(item)
    best_fit.put(item)

print('Next fit:    ', next_fit.get_bin_count())
print('First fit:   ', first_fit.get_bin_count())
print('Best fit:    ', best_fit.get_bin_count())
print('Learned fit: ', env.bin_count)

x = [i+1 for i in range(n_games)]
plot_learning(x, scores, eps_history, "%s_plot.png" % filename)

