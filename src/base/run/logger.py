from abc import ABC, abstractmethod


class Logger(ABC):
    def __init__(self):
        self.iteration = 0
        self.iteration_vars = {}
        self.vars = {**self.iteration_vars}
        self.logs = []
        
        self.log({'iteration': self.iteration})
    
    # Should be called after every pass
    def update(self):
        self.logs.append(self.iteration_vars)
        self.iteration_vars = {}
        self.iteration += 1
        
        self.log({'iteration': self.iteration})
    
    def log(self, vars: dict):
        self.iteration_vars = {**self.iteration_vars, **vars}
        self.vars = {**self.vars, **vars}
    
    @abstractmethod
    def display_current(self):
        pass
    
    def state_dict(self):
        return {
            'logs': self.logs,
            'iteration': self.iteration,
            'iteration_vars': self.iteration_vars,
            'vars': self.vars
        }
    
    def load_state_dict(self, state_dict):
        self.logs = state_dict.pop('logs')
        self.iteration = state_dict.pop('iteration')
        self.iteration_vars = state_dict.pop('iteration_vars')
        self.vars = state_dict.pop('vars')


class PrintLogger(Logger):
    def display_current(self):
        print(self.vars)
