from random import random
from causallib.estimation import IPW, Matching, PropensityMatching, TMLE, Standardization, StratifiedStandardization, XLearner
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

class DataCIModel:

    def __init__(self, dataframe: pd.DataFrame, *args, **kwargs):
        self.dataframe = dataframe
        self.dataframe.reset_index(drop=True, inplace=True)
        self.X = self.dataframe.drop(columns=['T','Y'])
        self.Y = self.dataframe['Y']
        self.T = self.dataframe['T']

    def calc_ATE(self, model, *args, **kwargs):
        model.fit(self.X, self.T)
        potential_outcomes = model.estimate_population_outcome(self.X, self.T, self.Y)
        treatments = self.T.unique()
        ATE_matrix = pd.DataFrame(index=treatments, columns=treatments)
        for t1 in treatments:
            for t2 in treatments:
                if t1 != t2:
                    ATE_matrix.loc[t1, t2] = model.estimate_effect(potential_outcomes[t1], potential_outcomes[t2]).values[0]
        return ATE_matrix

class CausalInferenceEstimationModel(ABC):

    def __init__(self, model_name: str, *args, **kwargs):
        self.model_name = model_name
        self.model = None

    @abstractmethod
    def estimate_ATE(self, data: DataCIModel, *args, **kwargs):
        pass

    def clac_ATE_matrix_from_po(self, data: DataCIModel, potential_outcomes):
        treatments = data.T.unique()
        ATE_matrix = pd.DataFrame(index=treatments, columns=treatments)
        for t1 in treatments:
            for t2 in treatments:
                if t1 != t2:
                    ATE_matrix.loc[t1, t2] = \
                        self.model.estimate_effect(potential_outcomes[t1], potential_outcomes[t2]).values[0]
                else:
                    ATE_matrix.loc[t1, t2] = 0
        return ATE_matrix

    def ind(self, data: DataCIModel):
        self.model.fit(data.X, data.T, data.Y)
        return self.model.estimate_individual_outcome(data.X, data.T)

class IPWlr(CausalInferenceEstimationModel):

    def __init__(self, lr=LogisticRegression(max_iter=10000, random_state=42), *args, **kwargs):
        super().__init__("IPWLogisticRegression",*args, **kwargs)
        self.model = IPW(lr)

    def estimate_ATE(self, data: DataCIModel, *args, **kwargs):
        self.model.fit(data.X, data.T)
        potential_outcomes = self.model.estimate_population_outcome(data.X, data.T, data.Y)
        ATE_matrix = self.clac_ATE_matrix_from_po(data, potential_outcomes)
        return ATE_matrix

class MatchingModel(CausalInferenceEstimationModel):

    def __init__(self, metric, n=1, *args, **kwargs):
        super().__init__("Matching", *args, **kwargs)
        self.model = Matching(metric=metric)

    def estimate_ATE(self, data: DataCIModel, *args, **kwargs):
        self.model.fit(data.X, data.T, data.Y)
        potential_outcomes = self.model.estimate_population_outcome(data.X, data.T)
        ATE_matrix = self.clac_ATE_matrix_from_po(data, potential_outcomes)
        return ATE_matrix

class PropensityMatchingModel(CausalInferenceEstimationModel):

    def __init__(self, learner, *args, **kwargs):
        super().__init__("PropensityMatching", *args, **kwargs)
        self.model = PropensityMatching(learner=learner)

    def estimate_ATE(self, data: DataCIModel, *args, **kwargs):
        self.model.fit(data.X, data.T, data.Y)
        potential_outcomes = self.model.estimate_population_outcome(data.X, data.T)
        ATE_matrix = self.clac_ATE_matrix_from_po(data, potential_outcomes)
        return ATE_matrix

class StandardizationModel(CausalInferenceEstimationModel):

    def __init__(self, lm, *args, **kwargs):
        super().__init__("Standardization", *args, **kwargs)
        self.model = Standardization(lm)

    def estimate_ATE(self, data: DataCIModel, *args, **kwargs):
        self.model.fit(data.X, data.T, data.Y)
        potential_outcomes = self.model.estimate_population_outcome(data.X, data.T)
        ATE_matrix = self.clac_ATE_matrix_from_po(data, potential_outcomes)
        return ATE_matrix

class TMLEModel(CausalInferenceEstimationModel):

    def __init__(self, outcome_model, weight_model, *args, **kwargs):
        super().__init__("TMLE", *args, **kwargs)
        self.model = TMLE(outcome_model = outcome_model, weight_model=weight_model, reduced=False)

    def estimate_ATE(self, data: DataCIModel, *args, **kwargs):
        self.model.fit(data.X, a=data.T, y=data.Y)
        potential_outcomes = self.model.estimate_population_outcome(data.X, a=data.T)
        ATE_matrix = self.clac_ATE_matrix_from_po(data, potential_outcomes)
        return ATE_matrix

class CausalInferenceEstimations:

    def __init__(self, *args, **kwargs):

        self.dict_of_estimation_models = dict()

        # IPW with logistic regression
        # self.dict_of_estimation_models['IPW_lr_lbfgs_10000'] = IPWlr()
        self.dict_of_estimation_models['IPW_lr_l1_saga_10000'] = IPWlr(lr=LogisticRegression(penalty='l1', solver="saga", max_iter=10000, random_state=42))

        #IPW with Boosting Classifier
        self.dict_of_estimation_models['IPW_Boosting'] = IPWlr(lr=GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42))

        # Matching with Euclidean distance
        self.dict_of_estimation_models['Matching_Euclidean'] = MatchingModel(metric='euclidean')

        # Matching with Mahalanobis distance
        self.dict_of_estimation_models['Matching_Mahalanobis'] = MatchingModel(metric='mahalanobis')

        # TMLE with IPW and Lasso
        self.dict_of_estimation_models['TMLE_Lasso'] = TMLEModel(outcome_model=Standardization(Lasso(random_state=42)), weight_model=IPW(GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)))

        # TMLE with complex models
        # https://github.com/BiomedSciAI/causallib/blob/master/examples/TMLE.ipynb
        outcome_model = Standardization(make_pipeline(PolynomialFeatures(2), LassoCV(random_state=0)))
        weight_model = IPW(make_pipeline(PolynomialFeatures(2), GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)))
        self.dict_of_estimation_models['TMLE_complex'] = TMLEModel(outcome_model=outcome_model, weight_model=weight_model)

        # Propensity Matching with GradientBoostingClassifier
        self.dict_of_estimation_models['PropensityMatching_lr'] = PropensityMatchingModel(learner=GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42))

        # Standardization with linear regression
        self.dict_of_estimation_models['Standardization_linear'] = StandardizationModel(LinearRegression())

        # Standardization with GradientBoostingRegressor
        self.dict_of_estimation_models['Standardization_Boosting'] = StandardizationModel(GradientBoostingRegressor())

    def estimate_with_all_models(self, data: DataCIModel):
        ATE_dict = dict()
        for model_name, model in self.dict_of_estimation_models.items():
            ATE = model.estimate_ATE(data)
            ATE_dict[model_name] = ATE
        return ATE_dict

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

        plt.savefig(f"/Users/danmizrahi/Desktop/causalInference/CausalInferenceProject/causal_inference_models/Figures/ATE_graph_T={ATE_index[0]}_T={ATE_index[1]}_trained_on_{data_set_name}.png")

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
        errors.append((abs(ate - ate_hat)/abs(ate)))
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
    plt.savefig(f"/Users/danmizrahi/Desktop/causalInference/CausalInferenceProject/causal_inference_models/Figures/L1_error_{data_set_name}.png")
    plt.clf()
    sorted_keys = sorted(dict_of_values_re.keys(), key=lambda x: dict_of_values_re[x])
    sorted_values = [dict_of_values_re[key] for key in sorted_keys]
    plt.scatter(sorted_keys, sorted_values, marker='x', s=60)
    plt.title("Relative Error for each model on " + data_set_name, fontsize=18)
    plt.xticks(rotation=35)
    plt.savefig(f"/Users/danmizrahi/Desktop/causalInference/CausalInferenceProject/causal_inference_models/Figures/Relative_error_{data_set_name}.png")

if __name__ == "__main__":
    df = pd.read_csv("/Users/danmizrahi/Desktop/causalInference/CausalInferenceProject/Simulated_Data/Training_data_3000.csv", index_col=0)
    df = pd.get_dummies(df, dtype=float)

    cols_to_scale = ['b_mean_E', 'b_mean_N', 'b_mean_S', 'b_mean_W', 'b_std_E',
                     'b_std_N', 'b_std_S', 'b_std_W', 'd_E', 'd_N', 'd_S']
    scale = MinMaxScaler()
    df[cols_to_scale] = scale.fit_transform(df[cols_to_scale])

    data = DataCIModel(df)

    causal_inference_estimations = CausalInferenceEstimations()

    ATE_dict = causal_inference_estimations.estimate_with_all_models(data)

    # Get true ATEs
    true_ATEs = pd.read_csv("/Users/danmizrahi/Desktop/causalInference/CausalInferenceProject/Simulated_Data/testing_ATEs.csv", index_col=0)
    # Get test data
    # Test whole data
    test_data = pd.read_csv(
        "/Users/danmizrahi/Desktop/causalInference/CausalInferenceProject/Simulated_Data/Testing_data_3000.csv",
        index_col=0)

    for idx in [[0,1], [0,2], [1,2]]:
         causal_inference_estimations.graph_ATE(ATE_dict, idx, true_ATEs.iloc[idx[0], idx[1]], data_set_name="Train_3000")

    l1_re_graphs(causal_inference_estimations, data, test_data, "Train_3000")

    lr = LogisticRegression(max_iter=10000, random_state=42)
    lr.fit(data.X, data.T)
    df['propensity_score_lr'] = lr.predict_proba(data.X)[:, 1]
    filtered_df = df.loc[df['propensity_score_lr'] > 0.1]
    filtered_df = filtered_df.loc[filtered_df['propensity_score_lr'] < 0.4]
    data = DataCIModel(filtered_df)
    causal_inference_estimations = CausalInferenceEstimations()
    ATE_dict = causal_inference_estimations.estimate_with_all_models(data)

    for idx in [[0,1], [0,2], [1,2]]:
            causal_inference_estimations.graph_ATE(ATE_dict, idx, true_ATEs.iloc[idx[0], idx[1]], data_set_name="Train_3000_filtered_by_overlap")

    l1_re_graphs(causal_inference_estimations, data, test_data, "Train_3000_filtered_by_overlap")
    df = pd.read_csv("/Users/danmizrahi/Desktop/causalInference/CausalInferenceProject/Simulated_Data/Training_data.csv",
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