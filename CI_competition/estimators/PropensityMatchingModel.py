import os

from .CausalInferenceEstimationModel import CausalInferenceEstimationModel
from causallib.estimation import PropensityMatching
from CI_competition.data.DataCIModel import DataCIModel
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier
class PropensityMatchingModel(CausalInferenceEstimationModel):

    def __init__(self,
                 learner:BaseEstimator = GradientBoostingClassifier(),
                 *args, **kwargs):
        """
        Propensity Matching Model
        :param learner: a scikit-learn classifier
        """
        super().__init__("PropensityMatching", *args, **kwargs)
        self.model = PropensityMatching(learner=learner)

    def estimate_ATE(self, data: DataCIModel, *args, **kwargs):
        self.model.fit(data.X, data.T, data.Y)
        potential_outcomes = self.model.estimate_population_outcome(data.X, data.T)
        ATE_matrix = self.calc_ATE_matrix_from_po(data, potential_outcomes)
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ATE_matrix.to_pickle(f"{parent_dir}/ATEs/{self.__str__()}.pkl")
        return ATE_matrix

    def __str__(self):
        return f"PropensityMatching_{self.model.learner.__str__()}".replace("()", "")
