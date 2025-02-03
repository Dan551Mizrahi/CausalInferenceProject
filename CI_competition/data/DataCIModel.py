import pandas as pd
class DataCIModel:

    def __init__(self, dataframe: pd.DataFrame, *args, **kwargs):
        self.dataframe = dataframe
        self.dataframe.reset_index(drop=True, inplace=True)
        self.X = self.dataframe.drop(columns=['T', 'Y'])
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
                    ATE_matrix.loc[t1, t2] = \
                        model.estimate_effect(potential_outcomes[t1], potential_outcomes[t2]).values[0]
        return ATE_matrix
