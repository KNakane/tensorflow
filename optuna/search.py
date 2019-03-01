#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append('./utility')
import optuna
import tensorflow as tf
from utils import Utils
from functools import partial

class Optuna():
    def __init__(self, db_name=None):
        self.name = db_name if db_name is not None else 'example-study'

    def search(self, obj, trials):
        assert trials > 0, "trial is bigger than 0"
        util = Utils(prefix='optuna')
        util.conf_log()
        study = optuna.create_study(study_name=self.name, storage='sqlite:///{}/hypara_search.db'.format(util.res_dir))
        study.optimize(obj, n_trials=trials)
        return

    def confirm(self, directory):
        self.study = optuna.Study(study_name=self.name, storage='sqlite:///{}/hypara_search.db'.format(directory))
        self.df = self.study.trials_dataframe()
        """
        self.study.best_params  # Get best parameters for the objective function.
        self.study.best_value  # Get best objective value.
        self.study.best_trial  # Get best trial's information.
        self.study.trials  # Get all trials' information.
        """