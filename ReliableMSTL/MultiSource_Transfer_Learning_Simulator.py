__author__ = "Zirui Wang"

"""
	Description:
		A simple simulator to demonstrate how to use PW-MSTL method
		experiment parameters can be tuned below
"""

from config_MSTL import *


def MultiSource_Transfer_Learning_Simulator(exp_params):
    # k sources, each with n instances with d dimensions
    k, n, d = exp_params["k"], exp_params["n"], exp_params["d"]
    X_source, Y_source, source_labeled_indices = [], [], []
    for i in range(k):
        X_source.append(np.random.rand(n, d))
        Y_source.append(np.random.choice([-1, 1], n).reshape(-1, 1))
        source_labeled_indices.append(np.random.choice(range(X_source[i].shape[0]), int(n / 2), replace=False))
    X_target_train = np.random.rand(n, d)
    X_target_test, Y_target_test = np.random.rand(n, d), np.random.choice([-1, 1], n).reshape(-1, 1)

    # Initialize the model with model parameters
    params = exp_params2model_params(exp_params)
    model = ReliableMultiSourceModel(X_source, Y_source, X_target_train, source_labeled_indices, params)
    model.train_models(exp_params["base_model"])

    Y_predicted = []
    for i in range(X_target_test.shape[0]):
        Y_predicted.append(model.multi_source_classify_PWMSTL(X_target_test[i].reshape(1, -1)))
    print("Accuracy: ", accuracy_score(Y_predicted, Y_target_test))


# Experimental Parameters
if __name__ == '__main__':
    exp_params = {"start_mode": "warm", "b1": 1.0, "tau_lambda": 1.0, "rho": 1.0, "beta_1": 10.0, "beta_2": 10.0,
                  "mu": 0.1, "max_alpha": 10.0, "AL_method": "DUMMY", "base_model": "svm", "k": 10, "n": 100, "d": 10}

    # Core parameters required for PW-MSTL

    # Dummy parameters

    # Experiment specific parameters

    MultiSource_Transfer_Learning_Simulator(exp_params)
