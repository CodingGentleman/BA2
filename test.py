from environment import BinEnvironment
import csv
import numpy as np
import matplotlib.pyplot as plt
import decimal
import scipy
from scipy.stats import truncnorm

def play():
    max_simultaneously_bins = 5
    env = BinEnvironment(max_simultaneously_bins)
    print('Moves: 0(Left), 1(Right), 2(Kick), 3(Put)')
    done = False
    print(env.reset())
    score = 0
    while not done:
        env.render()
        observation_, reward, done, info = env.step(int(input()))
        print(observation_, reward)
        score += reward
    print('final score %i' % score)

def test_csv():
    results = np.array([])
    with open("impl/input.csv") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            results = np.append(results, row)
    results = results.astype(np.float32)
    return results

def gen_trainingsdata():
    standard_defiation = 10000
    trainingdata_size = 10000
    x = get_truncated_normal(mean=50, standard_defiation=standard_defiation, lower_limit=1, upper_limit=99)
    data = (x.rvs(trainingdata_size)/100).round(decimals=2)
    # x2 = get_truncated_normal(mean=75, standard_defiation=standard_defiation, lower_limit=50, upper_limit=99)
    # data2 = (x2.rvs(5000)/100).round(decimals=2)
    # data = np.hstack([data,data2]).transpose()
    np.savetxt("training_data/flat.csv", data, delimiter=",", fmt='%.2f', newline=",")
    plt.hist(data, bins=10)
    plt.ylabel("Anzahl")
    plt.xlabel("Wert")
    plt.savefig("training_data/flat.png")

def get_truncated_normal(mean=0, standard_defiation=1, lower_limit=0, upper_limit=10):
    return truncnorm((lower_limit - mean) / standard_defiation, (upper_limit - mean) / standard_defiation, loc=mean, scale=standard_defiation)

if __name__ == "__main__":
    gen_trainingsdata()
