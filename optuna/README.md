Optuna
====

# Overview
Optunaを使用してはハイパーパラメータサーチを行う  

# Description
PFNが18年12月に発表したハイパーパラメータ自動最適化フレームワークをtensorflowに応用してlearning rate等のハイパーパラメータを最適化する

- [公式ページ](https://optuna.org/)
- [公式ドキュメント](https://optuna.readthedocs.io/en/stable/)
- [GitHub](https://github.com/pfnet/optuna)

# Usage
- search.py内にあるOptunaを呼び、search関数で最適化を行う
- 計算終了後、作成されたdbをconfirm関数で呼び出すことで結果を見ることが出来る
```python
from search import Optuna

op = Optuna('example-study')    # databaseの名前
op.search(obj, para, trials))   # obj:tensorflowのプログラム
                                # para:最適化したいパラメータ
                                # trials:searchする回数
op.confirm('results')           # directory名
op.study.best_params            # Get best parameters for the objective function.
op.study.best_value             # Get best objective value.
op.study.best_trial             # Get best trial's information.
op.study.trials                 # Get all trials' information.
```