from random import random
from causallib.estimation import IPW, Matching, PropensityMatching, TMLE, Standardization
from sklearn.linear_model import LogisticRegression, Lasso, Ridge, LinearRegression, LassoCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from abc import ABC, abstractmethod

from wandb.sklearn.plot.classifier import classifier

solver = "saga"

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

class IPWlr(CausalInferenceEstimationModel):

    def __init__(self, lr=LogisticRegression(solver=solver, max_iter=1000), *args, **kwargs):
        super().__init__("IPWLogisticRegression",*args, **kwargs)
        self.model = IPW(lr)

    def estimate_ATE(self, data: DataCIModel, *args, **kwargs):
        self.model.fit(data.X, data.T)
        potential_outcomes = self.model.estimate_population_outcome(data.X, data.T, data.Y)
        ATE_matrix = self.clac_ATE_matrix_from_po(data, potential_outcomes)
        return ATE_matrix

class MatchingModel(CausalInferenceEstimationModel):

    def __init__(self, metric, *args, **kwargs):
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
        self.dict_of_estimation_models['IPW_lr_lbfgs_1000'] = IPWlr()
        self.dict_of_estimation_models['IPW_lr_l1_saga_1000'] = IPWlr(lr=LogisticRegression(penalty='l1', solver=solver, max_iter=1000))

        #IPW with Boosting Classifier
        self.dict_of_estimation_models['IPW_Boosting'] = IPWlr(lr=GradientBoostingClassifier())

        # Matching with Euclidean distance
        self.dict_of_estimation_models['Matching_Euclidean'] = MatchingModel(metric='euclidean')

        # Matching with Mahalanobis distance
        self.dict_of_estimation_models['Matching_Mahalanobis'] = MatchingModel(metric='mahalanobis')

        # TMLE with IPW and Lasso
        self.dict_of_estimation_models['TMLE_Lasso'] = TMLEModel(outcome_model=Standardization(Lasso(random_state=42)), weight_model=IPW(LogisticRegression(penalty="l1", solver=solver, random_state=42)))

        # TMLE with complex models
        # https://github.com/BiomedSciAI/causallib/blob/master/examples/TMLE.ipynb
        outcome_model = Standardization(make_pipeline(PolynomialFeatures(2), LassoCV(random_state=0)))
        weight_model = IPW(make_pipeline(PolynomialFeatures(2), LogisticRegression(penalty="l1", C=0.00001, solver=solver)))
        self.dict_of_estimation_models['TMLE_complex'] = TMLEModel(outcome_model=outcome_model, weight_model=weight_model)

        # Propensity Matching with logistic regression
        self.dict_of_estimation_models['PropensityMatching_lr'] = PropensityMatchingModel(learner=LogisticRegression(solver=solver, max_iter=1000))

        # Standardization with linear regression
        self.dict_of_estimation_models['Standardization_lr'] = StandardizationModel(LinearRegression())

        # Standardization with GradientBoostingRegressor
        self.dict_of_estimation_models['Standardization_Boosting'] = StandardizationModel(GradientBoostingRegressor())


    def estimate_with_all_models(self, data: DataCIModel):
        ATE_dict = dict()
        for model_name, model in self.dict_of_estimation_models.items():
            ATE = model.estimate_ATE(data)
            ATE_dict[model_name] = ATE
        return ATE_dict

    def graph_ATE(self, ATE_dict: dict, ATE_index: tuple[int, int], true_ATE, *args, **kwargs):
        """Create a graph that shows the ATE for each model."""
        new_dict = {key: value.iloc[ATE_index[0], ATE_index[1]] for key, value in ATE_dict.items()}
        fig, ax = plt.subplots()
        fig.set_size_inches(30, 15)
        ax.bar(new_dict.keys(), new_dict.values())
        ax.xaxis.set_major_locator(FixedLocator(list(range(len(new_dict.keys())))))
        ax.set_xticklabels(new_dict.keys(), rotation=45)
        fig.suptitle(f"ATE for T={ATE_index[0]} and T={ATE_index[1]}")
        ax.set_ylabel("ATE")
        ax.set_xlabel("Model")

        # Add a line of true ATE
        ax.axhline(y=true_ATE, color='r', linestyle='-', label='True ATE')

        plt.savefig(f"ATE_graph_T={ATE_index[0]}_T={ATE_index[1]}.png")

    def latex_table_of_ATE(self, ATE_dict: dict,ATE_index: tuple[int, int], *args, **kwargs):
        """Create a LaTeX table that shows the ATE for each model."""
        new_dict = []#TODO
        table = pd.DataFrame(new_dict)
        table.to_latex()
        return table

if __name__ == "__main__":
    df = pd.read_csv("/Users/danmizrahi/Desktop/causalInference/CausalInferenceProject/Simulation/Training_data_3000.csv", index_col=0)
    df = pd.get_dummies(df, dtype=float)

    cols_to_scale = ['Prior', 'b_mean_E', 'b_mean_N', 'b_mean_S', 'b_mean_W', 'b_std_E',
                     'b_std_N', 'b_std_S', 'b_std_W', 'd_E', 'd_N', 'd_S']
    scale = MinMaxScaler()

    df[cols_to_scale] = scale.fit_transform(df[cols_to_scale])

    data = DataCIModel(df)

    causal_inference_estimations = CausalInferenceEstimations()

    ATE_dict = causal_inference_estimations.estimate_with_all_models(data)
    print(ATE_dict)
    # Get true ATEs
    true_ATEs = pd.read_csv("/Users/danmizrahi/Desktop/causalInference/CausalInferenceProject/Simulation/Testing_ATEs_3000.csv", index_col=0)

    for idx in [[0,1], [0,2], [1,2]]:
         causal_inference_estimations.graph_ATE(ATE_dict, idx, true_ATEs.iloc[idx[0], idx[1]])
