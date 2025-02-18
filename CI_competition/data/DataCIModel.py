import pandas as pd
from sklearn.preprocessing import MinMaxScaler





class DataCIModel:

    def __init__(self, dataframe: pd.DataFrame, *args, **kwargs):
        self.dataframe = dataframe
        self._preprocess_df()
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

    def _scale_df(self):
        """
        Scale the columns of the dataframe beside T, Prior and Y
        """
        cols = self.dataframe.columns
        cols_to_scale = [col for col in cols if col not in ["T", "Prior", "Y"]]
        scale = MinMaxScaler()
        self.dataframe[cols_to_scale] = scale.fit_transform(self.dataframe[cols_to_scale])

    def _preprocess_df(self):
        """
        Preprocess the dataframe
        """
        df = pd.get_dummies(self.dataframe, dtype=float)
        self.dataframe = df
        self._scale_df()
