import os

from sklearn.linear_model import LogisticRegression
from CI_competition.estimators.CausalInferenceEstimationModel import CausalInferenceEstimationModel
from causallib.estimation import IPW
from CI_competition.data.DataCIModel import DataCIModel
from sklearn.base import BaseEstimator

class IPWModel(CausalInferenceEstimationModel):

    def __init__(self,
                 learner: BaseEstimator = LogisticRegression(),
                 *args,
                 **kwargs):
        """
        IPW with Logistic Regression
        """
        super().__init__("IPWLogisticRegression", *args, **kwargs)
        self.model = IPW(learner=learner)

    def estimate_ATE(self, data: DataCIModel, *args, **kwargs):
        self.model.fit(data.X, data.T)
        potential_outcomes = self.model.estimate_population_outcome(data.X, data.T, data.Y)
        ATE_matrix = self.calc_ATE_matrix_from_po(data, potential_outcomes)
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ATE_matrix.to_pickle(f"{parent_dir}/ATEs/{self.__str__()}.pkl")
        return ATE_matrix

    def __str__(self):
        return f"IPW_{self.model.learner.__str__()}".replace("()", "")