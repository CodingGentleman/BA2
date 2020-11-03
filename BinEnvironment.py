import numpy as np
from decimal import Decimal
from item_generator import item_generator

class BinEnvironment():
    def __init__(self, max_simultaneously_bins, item_generator=item_generator()):
        self.max_simultaneously_bins = max_simultaneously_bins
        # possible actions: move left, move right, kick the bin, put the item into the bin
        self.action_space = {'L': -1, 'R': 1, 'K': 0, 'P': 0}
        self.available_actions = ['L', 'R', 'K', 'P']
        self.bin_count = 0
        self.item = 0
        self.item_generator = item_generator
        self.reset()

    def reset(self):
        self.bins = np.ones(self.max_simultaneously_bins, dtype=np.float32)
        self.agent_position = 0
        self.bin_count = 0
        self.item_generator.reset()
        self.item = self.item_generator.next()
        return [self.agent_position, self.item]

    def kick_bin(self):
        self.bins[self.agent_position] = np.float32(1)
        self.bin_count += 1

    def step(self, action):
        reward = -1
        terminal = not self.item_generator.has_next()
        chosen_action = self.available_actions[action]
        if chosen_action == 'P':
            bin_value = Decimal(str(self.bins[self.agent_position]))
            item_value = Decimal(str(self.item))
            if bin_value - item_value >= 0:
                self.bins[self.agent_position] = np.float32(str(bin_value - item_value))
                if not terminal:
                    self.item = self.item_generator.next()
                else:
                    reward = 100000
        elif chosen_action == 'K':
            self.kick_bin()
            reward = -10
        else:
            new_position = self.agent_position + self.action_space[chosen_action]
            if new_position >= 0 and new_position < self.max_simultaneously_bins:
                self.agent_position = new_position
                reward = 0
        return [self.agent_position, self.item], reward, terminal, None

    def render(self):
        print('------------------')
        print(self.bins)
        print("*".rjust(self.agent_position*2+2, ' '))
        print('------------------')

