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
        self.name = db_name if db_name is not None else 'example'
        self.util = Utils(prefix='optuna')
        self.util.conf_log()

    def search(self, obj, para, trials):
        assert trials > 0, "trial is bigger than 0"

        study = optuna.create_study(study_name=self.name, storage='sqlite:///{}/hypara_search.db'.format(self.util.res_dir))
        f = partial(obj, para)
        study.optimize(f, n_trials=trials)
        return

    def confirm(self, directory):
        study = optuna.Study(study_name=self.name, storage='sqlite:///{}/hypara_search.db'.format(directory))
        df = study.trials_dataframe()
        study.best_params  # Get best parameters for the objective function.
        study.best_value  # Get best objective value.
        study.best_trial  # Get best trial's information.
        study.trials  # Get all trials' information.