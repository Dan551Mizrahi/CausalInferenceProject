from results_reading import read_all_models, read_all_true_ates
import pandas as pd

def build_basic_ate_table(num_runs: int):
    """
    Create the ATE table, each row is a model (one row is the true ATE)
    and the columns are different treatments (T=1 and T=2), ATE is calculated against T=0.

    :param num_runs: The number of datasets
    :return: The ATE table
    """
    model_estimations = read_all_models(num_runs)
    true_ates = read_all_true_ates(num_runs)
    model_names = list(model_estimations.keys())
    ate_table = pd.DataFrame(columns=["T=1", "T=2"], index=model_names+["True ATE"])
    for model_name in model_names:
        # make an average matrix
        average_matrix = pd.concat(model_estimations[model_name]).astype('int64') .groupby(level=0).mean()
        # make a matrix of the 0.025 percentile
        lower_matrix = pd.concat(model_estimations[model_name]).astype('int64') .groupby(level=0).quantile(0.025)
        # make a matrix of the 0.975 percentile
        upper_matrix = pd.concat(model_estimations[model_name]).astype('int64') .groupby(level=0).quantile(0.975)
        # make a matrix of half of the length of the confidence interval
        ci_matrix = (upper_matrix - lower_matrix) / 2
        ate_table.loc[model_name, "T=1"] = str(round(average_matrix.loc[0, 1], 2)) + "±" + str(round(ci_matrix.loc[0, 1], 2))
        ate_table.loc[model_name, "T=2"] = str(round(average_matrix.loc[0, 2], 2)) + "±" + str(round(ci_matrix.loc[0, 2], 2))

    # add the true ATEs
    average_true_matrix = pd.concat(true_ates).astype('int64').groupby(level=0).mean()
    lower_true_matrix = pd.concat(true_ates).astype('int64').groupby(level=0).quantile(0.025)
    upper_true_matrix = pd.concat(true_ates).astype('int64').groupby(level=0).quantile(0.975)
    ci_true_matrix = (upper_true_matrix - lower_true_matrix) / 2
    ate_table.loc["True ATE", "T=1"] = str(round(average_true_matrix.iloc[0, 1], 2)) + "±" + str(round(ci_true_matrix.iloc[0, 1], 2))
    ate_table.loc["True ATE", "T=2"] = str(round(average_true_matrix.iloc[0, 2], 2)) + "±" + str(round(ci_true_matrix.iloc[0, 2], 2))

    return ate_table

def export_ate_table_csv(num_runs: int, filename: str):
    """
    Create and export the ATE table to a csv file
    :param num_runs: The number of datasets
    :param filename: The filename
    """
    ate_table = build_basic_ate_table(num_runs)
    ate_table.to_excel(filename)

if __name__ == "__main__":
    export_ate_table_csv(2, "basic_ate_table.xlsx")