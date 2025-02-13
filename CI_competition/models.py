from random import random
from causallib.estimation import IPW, TMLE, Standardization, StratifiedStandardization, \
    XLearner
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression, LassoCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="MinMaxScaler")

def graph_ATE(self, ATE_dict: dict, ATE_index: tuple[int, int], true_ATE, data_set_name="None", *args, **kwargs):
    """Create a graph that shows the ATE for each model."""
    new_dict = {key: value.loc[ATE_index[0], ATE_index[1]] for key, value in ATE_dict.items()}
    fig, ax = plt.subplots()
    fig.set_size_inches(30, 17)
    ax.bar(new_dict.keys(), new_dict.values())
    ax.xaxis.set_major_locator(FixedLocator(list(range(len(new_dict.keys())))))
    ax.set_xticklabels(new_dict.keys(), rotation=45)
    fig.suptitle(f"ATE for T={ATE_index[0]} and T={ATE_index[1]}", fontsize=18)
    ax.set_ylabel("ATE", fontsize=18)
    ax.set_xlabel("Model", fontsize=18)

    # Add a line of true ATE
    ax.axhline(y=true_ATE, color='r', linestyle='-', label='True ATE')

    plt.savefig(
            f"/Users/danmizrahi/Desktop/causalInference/CausalInferenceProject/causal_inference_models/Figures/ATE_graph_T={ATE_index[0]}_T={ATE_index[1]}_trained_on_{data_set_name}.png")


def find_index_in_true(idx: int):
    return (3 * idx, 3 * idx + 1, 3 * idx + 2)


def evaluate_l1_error(test_data, ind_outcomes, Treatments: tuple[int, int]):
    errors = []
    for idx in range(len(ind_outcomes)):
        idxs = find_index_in_true(idx)
        ate = test_data.iloc[idxs[Treatments[0]]]['Y'] - test_data.iloc[idxs[Treatments[1]]]['Y']
        ate_hat = ind_outcomes.iloc[idx, Treatments[0]] - ind_outcomes.iloc[idx, Treatments[1]]
        errors.append(abs(ate - ate_hat))
    return sum(errors) / len(errors)


def evaluate_relative_error(test_data, ind_outcomes, Treatments: tuple[int, int]):
    errors = []
    for idx in range(len(ind_outcomes)):
        idxs = find_index_in_true(idx)
        ate = test_data.iloc[idxs[Treatments[0]]]['Y'] - test_data.iloc[idxs[Treatments[1]]]['Y']
        ate_hat = ind_outcomes.iloc[idx, Treatments[0]] - ind_outcomes.iloc[idx, Treatments[1]]
        errors.append((abs(ate - ate_hat) / abs(ate)))
    return sum(errors) / len(errors)


def l1_re_graphs(causal_inference_estimations, data, test_data, data_set_name):
    plt.clf()
    # Evaluate the L1 and re error for each model amd for each pair of treatments
    dict_of_values_l1 = dict()
    dict_of_values_re = dict()
    for model_name, model in causal_inference_estimations.dict_of_estimation_models.items():
        try:
            ind_outcomes = model.ind(data)
            for i in range(1, 4):
                if i == 3:
                    l1_error = evaluate_l1_error(test_data, ind_outcomes, Treatments=(1, 2))
                    re_error = evaluate_relative_error(test_data, ind_outcomes, Treatments=(1, 2))
                    dict_of_values_l1[f"{model_name}_T=1_vs_T=2"] = l1_error
                    dict_of_values_re[f"{model_name}_T=1_vs_T=2"] = re_error
                else:
                    l1_error = evaluate_l1_error(test_data, ind_outcomes, Treatments=(0, i))
                    re_error = evaluate_relative_error(test_data, ind_outcomes, Treatments=(0, i))
                    dict_of_values_l1[f"{model_name}_T=0_vs_T={i}"] = l1_error
                    dict_of_values_re[f"{model_name}_T=0_vs_T={i}"] = re_error
        except:
            continue
    sorted_keys = sorted(dict_of_values_l1.keys(), key=lambda x: dict_of_values_l1[x])
    sorted_values = [dict_of_values_l1[key] for key in sorted_keys]
    plt.scatter(sorted_keys, sorted_values, marker='x', s=60)
    plt.title("L1 Error for each model on " + data_set_name, fontsize=18)
    plt.xticks(rotation=35)
    plt.savefig(
        f"/Users/danmizrahi/Desktop/causalInference/CausalInferenceProject/causal_inference_models/Figures/L1_error_{data_set_name}.png")
    plt.clf()
    sorted_keys = sorted(dict_of_values_re.keys(), key=lambda x: dict_of_values_re[x])
    sorted_values = [dict_of_values_re[key] for key in sorted_keys]
    plt.scatter(sorted_keys, sorted_values, marker='x', s=60)
    plt.title("Relative Error for each model on " + data_set_name, fontsize=18)
    plt.xticks(rotation=35)
    plt.savefig(
        f"/Users/danmizrahi/Desktop/causalInference/CausalInferenceProject/causal_inference_models/Figures/Relative_error_{data_set_name}.png")


if __name__ == "__main__":
    df = pd.read_csv(
        "/Users/danmizrahi/Desktop/causalInference/CausalInferenceProject/Simulated_Data/Training_data_3000.csv",
        index_col=0)
    df = pd.get_dummies(df, dtype=float)

    cols_to_scale = ['b_mean_E', 'b_mean_N', 'b_mean_S', 'b_mean_W', 'b_std_E',
                     'b_std_N', 'b_std_S', 'b_std_W', 'd_E', 'd_N', 'd_S']
    scale = MinMaxScaler()
    df[cols_to_scale] = scale.fit_transform(df[cols_to_scale])

    data = DataCIModel(df)

    causal_inference_estimations = CausalInferenceEstimations()

    ATE_dict = causal_inference_estimations.estimate_with_all_models(data)

    # Get true ATEs
    true_ATEs = pd.read_csv(
        "/Users/danmizrahi/Desktop/causalInference/CausalInferenceProject/Simulated_Data/testing_ATEs.csv", index_col=0)
    # Get test data
    # Test whole data
    test_data = pd.read_csv(
        "/Users/danmizrahi/Desktop/causalInference/CausalInferenceProject/Simulated_Data/Testing_data_3000.csv",
        index_col=0)

    for idx in [[0, 1], [0, 2], [1, 2]]:
        causal_inference_estimations.graph_ATE(ATE_dict, idx, true_ATEs.iloc[idx[0], idx[1]],
                                               data_set_name="Train_3000")

    l1_re_graphs(causal_inference_estimations, data, test_data, "Train_3000")

    lr = LogisticRegression(max_iter=10000, random_state=42)
    lr.fit(data.X, data.T)
    df['propensity_score_lr'] = lr.predict_proba(data.X)[:, 1]
    filtered_df = df.loc[df['propensity_score_lr'] > 0.1]
    filtered_df = filtered_df.loc[filtered_df['propensity_score_lr'] < 0.4]
    data = DataCIModel(filtered_df)
    causal_inference_estimations = CausalInferenceEstimations()
    ATE_dict = causal_inference_estimations.estimate_with_all_models(data)

    for idx in [[0, 1], [0, 2], [1, 2]]:
        causal_inference_estimations.graph_ATE(ATE_dict, idx, true_ATEs.iloc[idx[0], idx[1]],
                                               data_set_name="Train_3000_filtered_by_overlap")

    l1_re_graphs(causal_inference_estimations, data, test_data, "Train_3000_filtered_by_overlap")
    df = pd.read_csv(
        "/Users/danmizrahi/Desktop/causalInference/CausalInferenceProject/Simulated_Data/Training_data.csv",
        index_col=0)
    df = pd.get_dummies(df, dtype=float)

    cols_to_scale = ['b_mean_E', 'b_mean_N', 'b_mean_S', 'b_mean_W', 'b_std_E',
                     'b_std_N', 'b_std_S', 'b_std_W', 'd_E', 'd_N', 'd_S']
    scale = MinMaxScaler()
    df[cols_to_scale] = scale.fit_transform(df[cols_to_scale])

    data = DataCIModel(df)

    causal_inference_estimations = CausalInferenceEstimations()

    ATE_dict = causal_inference_estimations.estimate_with_all_models(data)

    for idx in [[0, 1], [0, 2], [1, 2]]:
        causal_inference_estimations.graph_ATE(ATE_dict, idx, true_ATEs.iloc[idx[0], idx[1]],
                                               data_set_name="Train_100")

    l1_re_graphs(causal_inference_estimations, data, test_data, "Train_100")
