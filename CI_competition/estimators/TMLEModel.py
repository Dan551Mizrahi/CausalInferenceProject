import os

from .CausalInferenceEstimationModel import CausalInferenceEstimationModel
from causallib.estimation import TMLE
from CI_competition.data.DataCIModel import DataCIModel
from causallib.estimation import Standardization
from causallib.estimation import IPW
class TMLEModel(CausalInferenceEstimationModel):

    def __init__(self,
                 outcome_model: Standardization,
                 weight_model: IPW,
                 *args, **kwargs):
        # TODO: verify that the outcome_model and weight_model are valid
        """
        Targeted Maximum Likelihood Estimation Model
        :param outcome_model: a CI estimation model for outcome model
        :param weight_model: a CI estimation model for weight model
        """
        super().__init__("TMLE", *args, **kwargs)
        self.model = TMLE(outcome_model=outcome_model, weight_model=weight_model, reduced=False)

    def estimate_ATE(self, data: DataCIModel, *args, **kwargs):
        self.model.fit(data.X, a=data.T, y=data.Y)
        potential_outcomes = self.model.estimate_population_outcome(data.X, a=data.T)
        ATE_matrix = self.calc_ATE_matrix_from_po(data, potential_outcomes)
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ATE_matrix.to_pickle(f"{parent_dir}/ATEs/{self.__str__()}.pkl")
        return ATE_matrix

    def __str__(self):
        return f"TMLE_{self.model.outcome_model.__str__()}_{self.model.weight_model.__str__()}".replace("()", "")