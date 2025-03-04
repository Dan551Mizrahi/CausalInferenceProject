from causallib.estimation import TMLE

from CI_competition.data.DataCIModel import DataCIModel
from CI_competition.estimators.IPWModel import IPWModel
from CI_competition.estimators.StandardizationModel import StandardizationModel
from .CausalInferenceEstimationModel import CausalInferenceEstimationModel


class TMLEModel(CausalInferenceEstimationModel):

    def __init__(self,
                 outcome_model: StandardizationModel,
                 weight_model: IPWModel,
                 *args, **kwargs):
        # TODO: verify that the outcome_model and weight_model are valid
        """
        Targeted Maximum Likelihood Estimation Model
        :param outcome_model: a CI estimation model for outcome model
        :param weight_model: a CI estimation model for weight model
        """
        super().__init__("TMLE", *args, **kwargs)
        self.outcome_model = outcome_model
        self.weight_model = weight_model
        self.model = TMLE(outcome_model=outcome_model.model, weight_model=weight_model.model, reduced=False)

    def estimate_ATE(self, data: DataCIModel, *args, **kwargs):
        self.model.fit(data.X, a=data.T, y=data.Y)
        potential_outcomes = self.model.estimate_population_outcome(data.X, a=data.T)
        ATE_matrix = self.calc_ATE_matrix_from_po(data, potential_outcomes)
        return ATE_matrix

    def __str__(self):
        return f"TMLE_{self.outcome_model.__str__()}_{self.weight_model.__str__()}".replace("()", "")
