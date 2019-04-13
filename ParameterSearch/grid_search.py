import os, sys
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

class Grid_Search():
    def __init__(self):
        self.param_grid = {}

    def search(self, obj, trials):
        grid = GridSearchCV(estimator=obj, param_grid=self.param_grid)
        return

class Random_Search():
    def __init__(self):
        self.param_grid = {}

    def search(self, obj, trials):
        grid = RandomizedSearchCV(estimator=obj, param_distributions=self.param_grid)
        return