import collections
import numpy as np
from decimal import Decimal

class abstract_base_class():
    def __init__(self, max_simultaneously_bins):
        self.bin_count = 0
        self.bin_size = np.float32("1")
        self.max_simultaneously_bins = max_simultaneously_bins
        self.last_bin_index = self.max_simultaneously_bins - 1
        self.bins = np.ones(self.max_simultaneously_bins, dtype=np.float32)
    
    def kick_bin(self):
        self.bins = np.where(self.bins==min(self.bins), np.float32("1"), self.bins)
        self.bin_count += 1

    def get_bin_count(self):
        return self.bin_count + len(self.bins) - collections.Counter(self.bins)[self.bin_size]
    
    def _convert_float_to_decimal(self, to_convert):
        return Decimal(str(to_convert))

class next_fit(abstract_base_class):
    def put(self, x):
        bin_value = Decimal(str(self.bins[0]))
        item_value = self._convert_float_to_decimal(x)
        if bin_value >= item_value: 
            self.bins[0] = np.float32(str(bin_value - item_value))
        else:
            self.kick_bin()
            self.bins[0] = np.float32(str(self._convert_float_to_decimal(self.bin_size) - item_value))

class first_fit(abstract_base_class):
    def put(self, x):
        item_value = self._convert_float_to_decimal(x)
        j = 0
        for j in range(self.max_simultaneously_bins):
            bin_value = Decimal(str(self.bins[j]))
            if bin_value >= item_value:
                self.bins[j] = np.float32(str(bin_value - item_value))
                return
        self.kick_bin()
        self.put(x)
        

class best_fit(abstract_base_class):
    def put(self, x):
        item_value = self._convert_float_to_decimal(x)
        min = Decimal("1")
        min_bin_index = 0
        for j in range(self.max_simultaneously_bins):
            bin_value = Decimal(str(self.bins[j]))
            if (bin_value >= item_value and bin_value - item_value < min): 
                min_bin_index = j
                min = bin_value - item_value
        if (min == Decimal("1")): 
            self.kick_bin()
            self.put(x)
        else: 
            bin_value = Decimal(str(self.bins[min_bin_index]))
            self.bins[min_bin_index] = np.float32(str(bin_value - item_value))



if __name__ == "__main__":
    from item_generator import item_generator
    max_simultaneously_bins = 3
    ig = item_generator(test_set=[Decimal('0.2'), Decimal('0.5'), Decimal('0.4'), Decimal('0.7'), Decimal('0.1'), Decimal('0.3'), Decimal('0.8')])
    next_fit = next_fit(max_simultaneously_bins)
    first_fit = first_fit(max_simultaneously_bins)
    best_fit = best_fit(max_simultaneously_bins)
    while ig.has_next():
        item = ig.next()
        print(item)
        next_fit.put(item)
        first_fit.put(item)
        best_fit.put(item)
    print('Next fit:  ', next_fit.get_bin_count())
    print('First fit: ', first_fit.get_bin_count())
    print('Best fit:  ', best_fit.get_bin_count())