import pandas as pd
import numpy as np
def calculate_ATEs(df):
    """
    :param df: A dataframe including the columns "T" and "Y"
    :return: a dataframe with the ATEs for each treatment
    """
    T_means = df.groupby("T")["Y"].mean()
    treatments = sorted(df["T"].unique().tolist())
    ATEs = {f"Baseline T={t}": [] for t in treatments}
    for t in treatments:
        for t2 in treatments:
            ATEs[f"Baseline T={t}"].append(T_means[t] - T_means[t2])
    df_ATEs = pd.DataFrame(ATEs, index=[f"Alternative T={t}" for t in treatments])
    return df_ATEs

def bootstrap_ATEs(df, num_samples=1000):
    """
    :param df: A dataframe including the columns "T" and "Y"
    :param num_samples: The number of bootstrap samples to take
    :return: a dataframe with the ATEs for each treatment, including CI bounds using bootstrap percentile method
    """
    df_ATEs = calculate_ATEs(df)
    bootstrap_samples = []
    for _ in range(num_samples):
        sample = df.sample(frac=1, replace=True)
        bootstrap_samples.append(calculate_ATEs(sample))

    # Stack the dataframes into a 3D numpy array
    stacked_data = np.dstack([df.values for df in bootstrap_samples])

    # Calculate the 0.025 and 0.975 percentiles along the third axis
    percentile_0025 = np.percentile(stacked_data, 2.5, axis=2)
    percentile_0975 = np.percentile(stacked_data, 97.5, axis=2)

    # Calculate half-length of the 95% confidence interval
    half_length_ci = (percentile_0975 - percentile_0025) / 2

    # Combine the mean and CI half-length into a single DataFrame with formatted strings
    ci_result_df = df_ATEs.round(3).astype(str) + " ± " + pd.DataFrame(half_length_ci, columns=df_ATEs.columns, index=df_ATEs.index).round(
        3).astype(str)
    return ci_result_df

if __name__ == '__main__':
    df = pd.read_csv("../Simulated_Data/Training_data.csv")
    print(bootstrap_ATEs(df))