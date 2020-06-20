__author__ = "Zirui Wang"

from config_MSTL import *


def exp_params2model_params(exp_params):
    params = {"AL_method": exp_params["AL_method"], "start_mode": exp_params["start_mode"], "b1": exp_params["b1"],
              "rho": exp_params["rho"], "beta_1": exp_params["beta_1"], "beta_2": exp_params["beta_2"],
              "mu": exp_params["mu"], "max_alpha": exp_params["max_alpha"], "tau_lambda": exp_params["tau_lambda"]}
    return params
