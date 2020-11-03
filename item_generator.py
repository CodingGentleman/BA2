import random
import numpy as np

class item_generator():
    def __init__(self, count=10, test_set=None):
        self.original_count = count
        self.count = count
        self.numbers = [np.float32('0.1'), np.float32('0.2'), np.float32('0.3'), np.float32('0.4'), np.float32('0.5'), np.float32('0.6'), np.float32('0.7'), np.float32('0.8'), np.float32('0.9')]
        self.weights = [10, 20, 30, 40, 50, 40, 30, 20, 10]
        self.test_set = test_set
        if self.test_set is not None:
            self.count = len(test_set)

    def reset(self):
        self.count = self.original_count

    def has_next(self):
        return self.count > 0

    def next(self):
        if self.test_set is not None:
            result = self.test_set[len(self.test_set)-self.count]
            self.count = self.count - 1
            return result
        if self.has_next():
            self.count -= 1
            print('count: %i' % self.count)
            return random.choices(self.numbers, weights=self.weights, k=1)[0]
        raise ValueError('No more items')

if __name__ == "__main__":
    ig = item_generator(count=2)
    print(ig.has_next())
    print(ig.next())
    print(ig.has_next())
    print(ig.next())
    print(ig.has_next())
    print(ig.next())
    print(ig.has_next())
    print(ig.next())
