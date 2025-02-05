from Figures.competition_figures.results_reading import read_all_models, read_all_true_ates
import pandas as pd
from error_functions import dict_of_error_functions
from agg_functions import dict_of_agg_functions

def build_basic_ate_table(num_runs: int, ci: bool = True):
    """
    Create the ATE table, each row is a model (one row is the true ATE)
    and the columns are different treatments (T=1 and T=2), ATE is calculated against T=0.

    :param num_runs: The number of datasets
    :param ci: Whether to include confidence intervals (95 percentile [0.025, 0.975])
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
        if ci:
            ate_table.loc[model_name, "T=1"] = "Mean: " + str(round(average_matrix.loc[0, 1], 2)) + "   95% quantile CI: [" + str(round(lower_matrix.loc[0, 1], 2)) + ", " + str(round(upper_matrix.loc[0, 1], 2)) + "]"
            ate_table.loc[model_name, "T=2"] = "Mean: " + str(round(average_matrix.loc[0, 2], 2)) + "   95% quantile CI: [" + str(round(lower_matrix.loc[0, 2], 2)) + ", " + str(round(upper_matrix.loc[0, 2], 2)) + "]"
        else:
            ate_table.loc[model_name, "T=1"] = round(average_matrix.loc[0, 1], 2)
            ate_table.loc[model_name, "T=2"] = round(average_matrix.loc[0, 2], 2)
    # add the true ATEs
    average_true_matrix = pd.concat(true_ates).astype('int64').groupby(level=0).mean()
    lower_true_matrix = pd.concat(true_ates).astype('int64').groupby(level=0).quantile(0.025)
    upper_true_matrix = pd.concat(true_ates).astype('int64').groupby(level=0).quantile(0.975)
    if ci:
        ate_table.loc["True ATE", "T=1"] = "Mean: " + str(round(average_true_matrix.iloc[0, 1], 2)) + "   95% quantile CI: [" + str(round(lower_true_matrix.iloc[0, 1], 2)) + ", " + str(round(upper_true_matrix.iloc[0, 1], 2)) + "]"
        ate_table.loc["True ATE", "T=2"] = "Mean: " + str(round(average_true_matrix.iloc[0, 2], 2)) + "   95% quantile CI: [" + str(round(lower_true_matrix.iloc[0, 2], 2)) + ", " + str(round(upper_true_matrix.iloc[0, 2], 2)) + "]"
    else:
        ate_table.loc["True ATE", "T=1"] = round(average_true_matrix.iloc[0, 1], 2)
        ate_table.loc["True ATE", "T=2"] = round(average_true_matrix.iloc[0, 2], 2)
    return ate_table


def build_paired_ate_table(num_runs: int, error_function: str = "relative_error", agg_function: str = "mean"):
    """
    Create the ATE table, each row is a model, we estimate the ATE for each treatment and then compare it to its true ATE.

    :param num_runs: The number of datasets
    :param error_function: The error function to use for calculating of the deviation from the true values
    :param agg_function: The aggregation function to use to aggragate the error values
    :return: The ATE table
    """
    error_function = dict_of_error_functions[error_function]
    agg_function = dict_of_agg_functions[agg_function]
    model_estimations = read_all_models(num_runs)
    true_ates = read_all_true_ates(num_runs)
    model_names = list(model_estimations.keys())
    ate_table = pd.DataFrame(columns=["T=1", "T=2"], index=model_names)
    dict_for_table = {(model_name, t): [] for model_name in model_names for t in ["T=1", "T=2"]}
    for i in range(len(true_ates)):
        true_ate = true_ates[i]
        for model_name in model_names:
            model_estimation = model_estimations[model_name][i]
            dict_for_table[(model_name, "T=1")].append(error_function(true_ate.iloc[0, 1], model_estimation.loc[0, 1]))
            dict_for_table[(model_name, "T=2")].append(error_function(true_ate.iloc[0, 2], model_estimation.loc[0, 2]))
    for model_name in model_names:
        ate_table.loc[model_name, "T=1"] = agg_function(dict_for_table[(model_name, "T=1")])
        ate_table.loc[model_name, "T=2"] = agg_function(dict_for_table[(model_name, "T=2")])

    return ate_table


def export_ate_table_excel(num_runs: int, filename: str, type: str = "basic", error_function: str = "relative_error", agg_function: str = "mean"):
    """
    Create and export the ATE table to an Excel file
    :param num_runs: The number of datasets
    :param filename: The filename
    :param type: The type of ATE table to create
    :param error_function: The error function to use for calculating of the deviation from the true values
    :param agg_function: The aggregation function to use to aggragate the error values
    """
    if type == "basic":
        ate_table = build_basic_ate_table(num_runs)
    elif type == "paired":
        ate_table = build_paired_ate_table(num_runs, error_function, agg_function)
    ate_table.to_excel(filename)

if __name__ == "__main__":
    export_ate_table_excel(2, "basic_ate_table.xlsx")
    export_ate_table_excel(2, "relative_error_ate_table.xlsx", type="paired", error_function="relative_error")
    export_ate_table_excel(2, "rmse_ate_table.xlsx", type="paired", error_function="squared_error", agg_function="rooted_mean")