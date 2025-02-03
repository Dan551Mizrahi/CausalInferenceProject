from CI_competition.estimators.IPWModel import IPWModel
from CI_competition.estimators.MatchingModel import MatchingModel
from CI_competition.estimators.PropensityMatchingModel import PropensityMatchingModel
from CI_competition.estimators.StandardizationModel import StandardizationModel
from CI_competition.estimators.TMLEModel import TMLEModel

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression



def models_definitions():
    return {
        "IPW":
            {
                "class": IPWModel,
                "params": [{"learner": LogisticRegression(solver="saga",penalty="l1")}]
            },
        "Matching":
            {
                "class": MatchingModel,
                "params": [{"metric": metric, "number_of_neighbours": number_of_neighbours}
                           for metric in ['euclidean', 'mahalanobis']
                           for number_of_neighbours in [1, 3, 5]]
            },
        "PropensityMatching":
            {
                "class": PropensityMatchingModel,
                "params": [{"learner": GradientBoostingClassifier()}]
            },
        "Standardization":
            {
                "class": StandardizationModel,
                "params": [{"learner": learner}
                           for learner in [LinearRegression(),GradientBoostingRegressor()]]
            },
        "TMLE":
            {
                "class": TMLEModel,
                "params": [{"outcome_model": StandardizationModel(LinearRegression()).model,
                            "weight_model": IPWModel(GradientBoostingClassifier()).model}]
            }
    }
