import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from CI_competition.utils import preprocess_df
from CI_competition.data.DataCIModel import DataCIModel
from CI_competition.estimators.models_definitions import models_definitions
from multiprocessing import Pool
from tqdm import tqdm


def calc_ate(args):
    model, data, ATEs_dir = args
    ATE_matrix = model.estimate_ATE(data)
    ATE_matrix.to_pickle(f"{ATEs_dir}/{model.__str__()}.pkl")


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

    # # Get true ATEs
    # true_ATEs = pd.read_csv(
    #     "/Users/danmizrahi/Desktop/causalInference/CausalInferenceProject/Simulated_Data/testing_ATEs.csv", index_col=0)
    # # Get test data
    # # Test whole data
    # test_data = pd.read_csv(
    #     "/Users/danmizrahi/Desktop/causalInference/CausalInferenceProject/Simulated_Data/Testing_data_3000.csv",
    #     index_col=0)
    #
    # for idx in [[0, 1], [0, 2], [1, 2]]:
    #     causal_inference_estimations.graph_ATE(ATE_dict, idx, true_ATEs.iloc[idx[0], idx[1]],
    #                                            data_set_name="Train_3000")
    #
    # l1_re_graphs(causal_inference_estimations, data, test_data, "Train_3000")
    #
    # lr = LogisticRegression(max_iter=10000, random_state=42)
    # lr.fit(data.X, data.T)
    # df['propensity_score_lr'] = lr.predict_proba(data.X)[:, 1]
    # filtered_df = df.loc[df['propensity_score_lr'] > 0.1]
    # filtered_df = filtered_df.loc[filtered_df['propensity_score_lr'] < 0.4]
    # data = DataCIModel(filtered_df)
    # causal_inference_estimations = CausalInferenceEstimations()
    # ATE_dict = causal_inference_estimations.estimate_with_all_models(data)
    #
    # for idx in [[0, 1], [0, 2], [1, 2]]:
    #     causal_inference_estimations.graph_ATE(ATE_dict, idx, true_ATEs.iloc[idx[0], idx[1]],
    #                                            data_set_name="Train_3000_filtered_by_overlap")
    #
    # l1_re_graphs(causal_inference_estimations, data, test_data, "Train_3000_filtered_by_overlap")
    # df = pd.read_csv(
    #     "/Users/danmizrahi/Desktop/causalInference/CausalInferenceProject/Simulated_Data/Training_data.csv",
    #     index_col=0)
    # df = pd.get_dummies(df, dtype=float)
    #
    # cols_to_scale = ['b_mean_E', 'b_mean_N', 'b_mean_S', 'b_mean_W', 'b_std_E',
    #                  'b_std_N', 'b_std_S', 'b_std_W', 'd_E', 'd_N', 'd_S']
    # scale = MinMaxScaler()
    # df[cols_to_scale] = scale.fit_transform(df[cols_to_scale])
    #
    # data = DataCIModel(df)
    #
    # causal_inference_estimations = CausalInferenceEstimations()
    #
    # ATE_dict = causal_inference_estimations.estimate_with_all_models(data)
    #
    # for idx in [[0, 1], [0, 2], [1, 2]]:
    #     causal_inference_estimations.graph_ATE(ATE_dict, idx, true_ATEs.iloc[idx[0], idx[1]],
    #                                            data_set_name="Train_100")
    #
    # l1_re_graphs(causal_inference_estimations, data, test_data, "Train_100")
