from abc import ABC, abstractmethod
import pandas as pd
from CI_competition.data.DataCIModel import DataCIModel


class CausalInferenceEstimationModel(ABC):
    def __init__(self, model_name: str, *args, **kwargs):
        self.model_name = model_name
        self.model = None

    @abstractmethod
    def estimate_ATE(self, data: DataCIModel, *args, **kwargs):
        pass

    def calc_ATE_matrix_from_po(self, data: DataCIModel, potential_outcomes):
        """
        Calculate the ATE matrix from the potential outcomes
        :param data: the training data
        :param potential_outcomes: the potential outcomes, expected result for a given treatment
        :return: a matrix of the ATE as a DataFrame
        """
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
