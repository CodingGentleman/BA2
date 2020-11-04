import numpy as np
from decimal import Decimal

class ItemProvider():
    def __init__(self, sample_size=10, data=[np.float32('0.1'), np.float32('0.2'), np.float32('0.3'), np.float32('0.4'), np.float32('0.5'), np.float32('0.6'), np.float32('0.7'), np.float32('0.8'), np.float32('0.9')], randomize=False):
        self.sample_size = sample_size
        self.counter = sample_size
        self.randomize = randomize
        self.data = data

    def reset(self):
        self.counter = self.sample_size

    def has_next(self):
        return self.counter > 0

    def next(self):
        if self.has_next():
            index = len(self.data)-self.counter
            if self.randomize:
                index = np.random.randint(0, len(self.data))
            self.counter -= 1
            return self.data[index]
        raise ValueError('No more items')

class BinEnvironment():
    def __init__(self, max_simultaneously_bins, item_provider=ItemProvider()):
        self.max_simultaneously_bins = max_simultaneously_bins
        # possible actions: move left, move right, kick the bin, put the item into the bin
        self.action_space = {'L': -1, 'R': 1, 'K': 0, 'P': 0}
        self.available_actions = ['L', 'R', 'K', 'P']
        self.bin_count = 0
        self.item = 0
        self.item_provider = item_provider
        self.reset()

    def reset(self):
        self.bins = np.ones(self.max_simultaneously_bins, dtype=np.float32)
        self.agent_position = 0
        self.bin_count = 0
        self.item_provider.reset()
        self.item = self.item_provider.next()
        return [self.agent_position, self.item]

    def kick_bin(self):
        self.bins[self.agent_position] = np.float32(1)
        self.bin_count += 1

    def step(self, action):
        reward = 0
        terminal = False
        chosen_action = self.available_actions[action]
        if chosen_action == 'P':
            bin_value = Decimal(str(self.bins[self.agent_position]))
            item_value = Decimal(str(self.item))
            if bin_value - item_value >= 0:
                self.bins[self.agent_position] = np.float32(str(bin_value - item_value))
                terminal = not self.item_provider.has_next()
                if not terminal:
                    self.item = self.item_provider.next()
        elif chosen_action == 'K':
            self.kick_bin()
            reward = -1
        else:
            new_position = self.agent_position + self.action_space[chosen_action]
            if new_position >= 0 and new_position < self.max_simultaneously_bins:
                self.agent_position = new_position
        return [self.agent_position, self.item], reward, terminal, None

    def render(self):
        print('------------------')
        print(self.bins)
        print('------------------')


if __name__ == "__main__":
    ig = ItemProvider(sample_size=2, data=[np.float32('0.1'), np.float32('0.2')], randomize=True)
    print(ig.has_next())
    print(ig.next())
    print(ig.has_next())
    print(ig.next())
    print(ig.has_next())
    print(ig.next())
    print(ig.has_next())
    print(ig.next())