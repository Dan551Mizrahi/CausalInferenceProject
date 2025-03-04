from causallib.estimation import Matching

from CI_competition.data.DataCIModel import DataCIModel
from .CausalInferenceEstimationModel import CausalInferenceEstimationModel


class MatchingModel(CausalInferenceEstimationModel):

    def __init__(self,
                 metric: str = 'mahalanobis',
                 number_of_neighbours: int = 1,
                 *args, **kwargs):
        """
        Matching Model
        :param metric: distance metric to use for matching. must be one of ['euclidean', 'mahalanobis']
        :param number_of_neighbours: number of neighbours to match
        """
        super().__init__("Matching", *args, **kwargs)
        self.model = Matching(metric=metric, n_neighbors=number_of_neighbours)

    def estimate_ATE(self, data: DataCIModel, *args, **kwargs):
        self.model.fit(data.X, data.T, data.Y)
        potential_outcomes = self.model.estimate_population_outcome(data.X, data.T)
        ATE_matrix = self.calc_ATE_matrix_from_po(data, potential_outcomes)
        return ATE_matrix

    def __str__(self):
        return f"Matching_{self.model.metric}_{self.model.n_neighbors}".replace("()", "")
