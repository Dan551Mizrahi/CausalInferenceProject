import os

from .CausalInferenceEstimationModel import CausalInferenceEstimationModel
from causallib.estimation import Standardization
from CI_competition.data.DataCIModel import DataCIModel
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression


class StandardizationModel(CausalInferenceEstimationModel):

    def __init__(self,
                 learner: BaseEstimator = LinearRegression(),
                 *args, **kwargs):
        """
        Standardization Model
        :param learner: a scikit-learn regressor
        """
        super().__init__("Standardization", *args, **kwargs)
        self.model = Standardization(learner=learner)

    def estimate_ATE(self, data: DataCIModel, *args, **kwargs):
        self.model.fit(data.X, data.T, data.Y)
        potential_outcomes = self.model.estimate_population_outcome(data.X, data.T)
        ATE_matrix = self.calc_ATE_matrix_from_po(data, potential_outcomes)
        return ATE_matrix

    def __str__(self):
        return f"Standardization_{self.model.learner.__str__()}".replace("()", "")
