""" This is a script which calls uses the runtime prediction model of OBOE.
    OBOE: https://github.com/udellgroup/oboe
"""
from sklearn.datasets import load_iris
from auto_learner import AutoLearner
import convex_opt

X, y = load_iris(return_X_y=True)
m = AutoLearner(runtime_limit=20)

# Produce runtime prediction based on dataset size:
t_predicted = convex_opt.predict_runtime(X.shape, runtime_matrix=m.runtime_matrix)
# zip(list(m.runtime_matrix.columns), t_predicted)  # relative(?) runtime per algorithm

# Create an ensemble of learners for dataset (lots of warnings):
m.fit(X, y)
# [el.algorithm for el in m.ensemble.base_learners]  # algorithms in ensemble
