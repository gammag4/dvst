from abc import ABC, abstractmethod


class Logger(ABC):
    def __init__(self):
        self._iteration = 0
        self._iteration_vars = {}
        self.vars = {**self._iteration_vars}
        self.logs = []
        
        self.log({'iteration': self._iteration})
    
    # Should be called after every pass
    def update(self):
        self.logs.append(self._iteration_vars)
        self._iteration_vars = {}
        self._iteration += 1
        
        self.log({'iteration': self._iteration})
    
    def log(self, vars: dict):
        self._iteration_vars = {**self._iteration_vars, **vars}
        self.vars = {**self.vars, **vars}
    
    @abstractmethod
    def display_current(self):
        pass
    
    def state_dict(self):
        return {
            'logs': self.logs,
            'iteration': self._iteration,
            'iteration_vars': self._iteration_vars,
            'vars': self.vars
        }
    
    def load_state_dict(self, state_dict):
        self.logs = state_dict.pop('logs')
        self._iteration = state_dict.pop('iteration')
        self._iteration_vars = state_dict.pop('iteration_vars')
        self.vars = state_dict.pop('vars')


class PrintLogger(Logger):
    def display_current(self):
        print(self.vars)
