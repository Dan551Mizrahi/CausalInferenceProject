import os
import pandas as pd
from CI_competition.utils import preprocess_df
from CI_competition.data.DataCIModel import DataCIModel
from CI_competition.estimators.models_definitions import models_definitions
from multiprocessing import Pool
from tqdm import tqdm


def calc_ate(args):
    model, data, ATEs_dir = args
    ATE_matrix = model.estimate_ATE(data)
    ATE_matrix.to_pickle(f"{ATEs_dir}/{model.__str__()}.pkl")


def calculate_true_ATEs(df, ATEs_dir):
    """
    :param df: A dataframe including the columns "T" and "Y"
    :return: a dataframe with the ATEs for each treatment
    """
    T_means = df.groupby("T")["Y"].mean()
    treatments = sorted(df["T"].unique().tolist())
    ATEs = {f"T={t}": [] for t in treatments}
    for t in treatments:
        for t2 in treatments:
            ATEs[f"T={t}"].append(T_means[t] - T_means[t2])
    df_ATEs = pd.DataFrame(ATEs, index=[f"T={t}" for t in treatments])
    df_ATEs.to_pickle(f"{ATEs_dir}/True.pkl")


def main(competition_args, run_args, training_df, index):
    curdir = os.path.dirname(__file__)
    ATEs_dir = os.path.join(curdir, "ATEs")
    os.makedirs(ATEs_dir, exist_ok=True)
    ATEs_dir = os.path.join(ATEs_dir, f"run_{index}")
    os.makedirs(ATEs_dir, exist_ok=True)

    # prepare data
    df = preprocess_df(training_df)
    data = DataCIModel(df)

    # Run all models
    MODELS_DEFINITIONS = models_definitions()
    if competition_args["model"]:
        models_instances = [MODELS_DEFINITIONS[competition_args["model"]]["class"](**params) for params in
                            MODELS_DEFINITIONS[competition_args["model"]]["params"]]
    else:
        models_instances = [MODELS_DEFINITIONS[model]["class"](**params) for model in MODELS_DEFINITIONS for params in
                            MODELS_DEFINITIONS[model]["params"]]

    with Pool(run_args["num_processes"]) as p:
        result = list(
            tqdm(p.map(calc_ate,
                       [(model, data, ATEs_dir) for model in models_instances]),
                 total=len(models_instances)))

    # Calculate the true ATEs
    calculate_true_ATEs(df, ATEs_dir)
