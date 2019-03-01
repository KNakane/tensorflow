#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import argparse
from search import Optuna


def main(args):
    op = Optuna('example-study')
    op.confirm(args.dir)
    print(op.study.best_params)
    print(op.study.best_value)
    print(op.study.best_trial)
    print(op.study.trials)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='please input optuna result dir')
    args = parser.parse_args()
    assert args.dir is not None
    main(args)