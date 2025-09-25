import time


class Timer:
    def __init__(self):
        self._n_deltas = 0
        self.delta = 0.0
        self.avg_delta = 0.0
        self._last_time = time.perf_counter()
    
    def eta(self, total_n):
        return total_n * self.avg_delta
    
    def update(self):
        t = time.perf_counter()
        self.delta = t - self._last_time
        self._last_time = t
        self.avg_delta = (self._n_deltas * self.avg_delta + self.delta) / (self._n_deltas + 1)
        
        self._n_deltas += 1
    
    def state_dict(self):
        return {
            'n_deltas': self._n_deltas,
            'delta': self.delta,
            'avg_delta': self.avg_delta,
            'last_time': self._last_time
        }
    
    def load_state_dict(self, state_dict):
        self._n_deltas = state_dict['n_deltas']
        self.delta = state_dict['delta']
        self.avg_delta = state_dict['avg_delta']
        self._last_time = state_dict['last_time']
