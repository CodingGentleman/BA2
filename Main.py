import numpy as np
import ClassicAlgorithms as ca
from BinEnvironment import BinEnvironment
from Agent import Agent
from Utils import plotLearning
from Utils import readCsv
from item_generator import item_generator

max_simultaneously_bins = 5
data = readCsv(file="training_data/micro_gauss.csv")
print(data)
ig = item_generator(count=len(data), test_set=data)

env = BinEnvironment(max_simultaneously_bins, item_generator=ig)
agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01, input_dims=[2], lr=0.001)

scores, eps_history = [], []
n_games = 500

for i in range(n_games):
    score = 0
    done = False
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
    print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score, 'epsilon %.2f' % agent.epsilon, 'bin count %i' % env.bin_count)

next_fit = ca.next_fit(max_simultaneously_bins)
first_fit = ca.first_fit(max_simultaneously_bins)
best_fit = ca.best_fit(max_simultaneously_bins)
ig.reset()
while ig.has_next():
    item = ig.next()
    next_fit.put(item)
    first_fit.put(item)
    best_fit.put(item)

print('Next fit:    ', next_fit.get_bin_count())
print('First fit:   ', first_fit.get_bin_count())
print('Best fit:    ', best_fit.get_bin_count())
print('Learned fit: ', env.bin_count)

x = [i+1 for i in range(n_games)]
plotLearning(x, scores, eps_history, 'plot.png')

