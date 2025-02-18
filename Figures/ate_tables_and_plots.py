from Figures.results_reading import read_all_models, read_all_true_ates
import pandas as pd
from Figures.error_functions import dict_of_error_functions
from Figures.agg_functions import dict_of_agg_functions
import matplotlib.pyplot as plt
import os

dict_of_short_model_names = {"TMLE_Standardization_LinearRegression_IPW_GradientBoostingClassifier": "TMLE S-Learner",
                             "PropensityMatching_GradientBoostingClassifier": "Propensity Matching",
                             "Matching_mahalanobis_5": "Matching Mal 5",
                             "Matching_mahalanobis_3": "Matching Mal 3",
                             "Matching_mahalanobis_1": "Matching Mal 1",
                             "Matching_euclidean_5": "Matching Euc 5",
                             "Matching_euclidean_3": "Matching Euc 3",
                             "Matching_euclidean_1": "Matching Euc 1",
                             "IPW_LogisticRegression(penalty='l1', solver='saga')": "IPW LR",
                             "Standardization_GradientBoostingRegressor": "S-Learner GBR",
                             "Standardization_LinearRegression": "S-Learner LR",}

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
    ate_table = pd.DataFrame(columns=["T=1", "T=2"], index=list(dict_of_short_model_names.values())+["True ATE"])
    for model_name in model_names:
        short_name = dict_of_short_model_names[model_name]
        # make an average matrix
        average_matrix = pd.concat(model_estimations[model_name]).astype('float64') .groupby(level=0).mean()
        # make a matrix of the 0.025 percentile
        lower_matrix = pd.concat(model_estimations[model_name]).astype('float64') .groupby(level=0).quantile(0.025)
        # make a matrix of the 0.975 percentile
        upper_matrix = pd.concat(model_estimations[model_name]).astype('float64') .groupby(level=0).quantile(0.975)
        if ci:
            ate_table.loc[short_name, "T=1"] = "Mean: " + str(round(average_matrix.loc[0, 1], 2)) + "   95% quantile CI: [" + str(round(lower_matrix.loc[0, 1], 2)) + ", " + str(round(upper_matrix.loc[0, 1], 2)) + "]"
            ate_table.loc[short_name, "T=2"] = "Mean: " + str(round(average_matrix.loc[0, 2], 2)) + "   95% quantile CI: [" + str(round(lower_matrix.loc[0, 2], 2)) + ", " + str(round(upper_matrix.loc[0, 2], 2)) + "]"
        else:
            ate_table.loc[short_name, "T=1"] = round(average_matrix.loc[0, 1], 2)
            ate_table.loc[short_name, "T=2"] = round(average_matrix.loc[0, 2], 2)
    # add the true ATEs
    average_true_matrix = pd.concat(true_ates).astype('float64').groupby(level=0).mean()
    lower_true_matrix = pd.concat(true_ates).astype('float64').groupby(level=0).quantile(0.025)
    upper_true_matrix = pd.concat(true_ates).astype('float64').groupby(level=0).quantile(0.975)
    if ci:
        ate_table.loc["True ATE", "T=1"] = "Mean: " + str(round(average_true_matrix.iloc[0, 1], 2)) + "   95% quantile CI: [" + str(round(lower_true_matrix.iloc[0, 1], 2)) + ", " + str(round(upper_true_matrix.iloc[0, 1], 2)) + "]"
        ate_table.loc["True ATE", "T=2"] = "Mean: " + str(round(average_true_matrix.iloc[0, 2], 2)) + "   95% quantile CI: [" + str(round(lower_true_matrix.iloc[0, 2], 2)) + ", " + str(round(upper_true_matrix.iloc[0, 2], 2)) + "]"
    else:
        ate_table.loc["True ATE", "T=1"] = round(average_true_matrix.iloc[0, 1], 2)
        ate_table.loc["True ATE", "T=2"] = round(average_true_matrix.iloc[0, 2], 2)
    return ate_table


def build_paired_ate_table(num_runs: int, error_function: str = "relative_error", agg_function: str = "Mean", ci: bool = False):
    """
    Create the ATE table, each row is a model, we estimate the ATE for each treatment and then compare it to its true ATE.

    :param num_runs: The number of datasets
    :param error_function: The error function to use for calculating of the deviation from the true values
    :param agg_function: The aggregation function to use to aggragate the error values
    :param ci: Whether to include confidence intervals (95 percentile [0.025, 0.975])
    :return: The ATE table
    """

    error_function = dict_of_error_functions[error_function]
    agg_function_string = agg_function
    agg_function = dict_of_agg_functions[agg_function]
    model_estimations = read_all_models(num_runs)
    true_ates = read_all_true_ates(num_runs)
    model_names = list(model_estimations.keys())
    ate_table = pd.DataFrame(columns=["T=1", "T=2"], index=list(dict_of_short_model_names.values()))
    dict_for_table = {(model_name, t): [] for model_name in model_names for t in ["T=1", "T=2"]}
    for i in range(len(true_ates)):
        true_ate = true_ates[i]
        for model_name in model_names:
            model_estimation = model_estimations[model_name][i]
            dict_for_table[(model_name, "T=1")].append(error_function(true_ate.iloc[0, 1], model_estimation.loc[0, 1]))
            dict_for_table[(model_name, "T=2")].append(error_function(true_ate.iloc[0, 2], model_estimation.loc[0, 2]))
    for model_name in model_names:
        short_name = dict_of_short_model_names[model_name]
        if ci:
            ate_table.loc[short_name, "T=1"] = f"{agg_function_string}: " + str(round(agg_function(dict_for_table[(model_name, "T=1")]), 2)) + "   95% quantile CI: [" + str(round(pd.Series(dict_for_table[(model_name, "T=1")]).quantile(0.025), 2)) + ", " + str(round(pd.Series(dict_for_table[(model_name, "T=1")]).quantile(0.975), 2)) + "]"
            ate_table.loc[short_name, "T=2"] = f"{agg_function_string}: " + str(round(agg_function(dict_for_table[(model_name, "T=2")]), 2)) + "   95% quantile CI: [" + str(round(pd.Series(dict_for_table[(model_name, "T=2")]).quantile(0.025), 2)) + ", " + str(round(pd.Series(dict_for_table[(model_name, "T=2")]).quantile(0.975), 2)) + "]"
        else:
            ate_table.loc[short_name, "T=1"] = round(agg_function(dict_for_table[(model_name, "T=1")]), 2)
            ate_table.loc[short_name, "T=2"] = round(agg_function(dict_for_table[(model_name, "T=1")]), 2)

    return ate_table


def export_ate_table_excel(num_runs: int, filename: str, type: str = "basic", error_function: str = "relative_error", agg_function: str = "Mean", **kwargs):
    """
    Create and export the ATE table to an Excel file
    :param num_runs: The number of datasets
    :param filename: The filename
    :param type: The type of ATE table to create
    :param error_function: The error function to use for calculating of the deviation from the true values
    :param agg_function: The aggregation function to use to aggragate the error values
    """
    if type == "basic":
        ate_table = build_basic_ate_table(num_runs, **kwargs)
    elif type == "paired":
        ate_table = build_paired_ate_table(num_runs, error_function, agg_function, **kwargs)
    ate_table.to_excel(filename)

def build_box_plots_graph(num_runs: int, save_path: str, trim_y_axis: bool = False):
    """
    Create the box plots graph, each box is a model's ATE over all the datasets and there is a red line representing the
    true ATE.
    :param num_runs: The number of datasets
    :param save_path: The path to save the plot
    :param trim_y_axis: Whether to trim the y-axis to 150% of the true ATE
    """
    model_estimations = read_all_models(num_runs)
    true_ates = read_all_true_ates(num_runs)
    model_names = list(dict_of_short_model_names.keys())
    shorted_model_names = [dict_of_short_model_names[model_name] for model_name in model_names]
    ate_t1 = pd.DataFrame(columns=[f"{i}" for i in range(len(true_ates))], index=shorted_model_names + ["True ATE"])
    ate_t2 = pd.DataFrame(columns=[f"{i}" for i in range(len(true_ates))], index=shorted_model_names + ["True ATE"])
    for model_name in model_names:
        for i in range(len(true_ates)):
            model_estimation = model_estimations[model_name][i]
            ate_t1.loc[dict_of_short_model_names[model_name], f"{i}"] = model_estimation.loc[0, 1]
            ate_t2.loc[dict_of_short_model_names[model_name], f"{i}"] = model_estimation.loc[0, 2]
            ate_t1.loc["True ATE", f"{i}"] = true_ates[i].iloc[0, 1]
            ate_t2.loc["True ATE", f"{i}"] = true_ates[i].iloc[0, 2]

    # Transpose the dataframes
    ate_t1 = ate_t1.T
    ate_t2 = ate_t2.T
    ate_t1 = ate_t1.astype('float')
    ate_t2 = ate_t2.astype('float')

    boxplot1 = ate_t1.boxplot()
    boxplot1.get_figure().set_size_inches(14,11)
    # boxplot1.set_title("Boxplot of ATE Estimations for T=1")
    boxplot1.set_ylabel("Estimated ATE for T=1")
    boxplot1.axhline(y=ate_t1["True ATE"].mean(), color='r', linestyle='--', label='True Mean ATE')
    boxplot1.tick_params(axis='x', rotation=45)
    boxplot1.legend()
    if trim_y_axis:
        boxplot1.set_ylim([2*ate_t1["True ATE"].mean(), ate_t1["True ATE"].mean()-1.5*ate_t1["True ATE"].mean()])
    path_to_save = os.path.join(save_path, "boxplot_T=1.pdf")
    boxplot1.get_figure().savefig(path_to_save)
    boxplot1.get_figure().clear()

    boxplot2 = ate_t2.boxplot()
    boxplot2.get_figure().set_size_inches(14, 11)
    # boxplot2.set_title("Boxplot of ATE Estimations for T=2")
    boxplot2.set_ylabel("Estimated ATE for T=2")
    boxplot2.axhline(y=ate_t2["True ATE"].mean(), color='r', linestyle='--', label='True Mean ATE')
    boxplot2.tick_params(axis='x', rotation=45)
    boxplot2.legend()
    if trim_y_axis:
        boxplot2.set_ylim([1.85*ate_t2["True ATE"].mean(), ate_t2["True ATE"].mean()-1.85*ate_t2["True ATE"].mean()])
    # Join paths
    path_to_save = os.path.join(save_path, "boxplot_T=2.pdf")
    boxplot2.get_figure().savefig(path_to_save)
    boxplot2.get_figure().clear()


def create_paired_graph(num_runs: int, model_name: str, save_path: str, **kwargs):
    """
    This function creates two scatter plots of the ATE estimations of a model against the true ATE.
    The x-axis is the ordinal number of the dataset and the y-axis is the ATE estimation.
    The function will draw a black vertical line between the true ATE (in green) and the model's ATE estimation (in purple) for each dataset.
    :param num_runs: The number of datasets
    :param model_name: The model name
    :param save_path: The path to save the plot
    """
    model_estimations = read_all_models(num_runs)
    true_ates = read_all_true_ates(num_runs)
    model_estimations = model_estimations[model_name]
    true_ates = true_ates
    model_name = dict_of_short_model_names[model_name]

    for i in range(len(true_ates)):
        true_ate = true_ates[i]
        model_estimation = model_estimations[i]
        plt.scatter([i], [true_ate.iloc[0, 1]], color='green')
        plt.scatter([i], [model_estimation.loc[0, 1]], color='purple')
        plt.plot([i, i], [true_ate.iloc[0, 1], model_estimation.loc[0, 1]], color='black')
    # plt.title(f"{model_name} - ATE Estimations for T=1")
    plt.ylabel("ATE")
    plt.xlabel("Dataset")
    plt.xticks([])
    plt.scatter([], [], color='green', label='True ATE')
    plt.scatter([], [], color='purple', label='Model ATE')
    plt.legend(loc = 'upper right')
    save_path1 = os.path.join(save_path, f"{model_name}_T1.pdf")
    plt.savefig(save_path1)
    plt.cla()
    plt.clf()

    for i in range(len(true_ates)):
        true_ate = true_ates[i]
        model_estimation = model_estimations[i]
        plt.scatter([i], [true_ate.iloc[0, 2]], color='green')
        plt.scatter([i], [model_estimation.loc[0, 2]], color='purple')
        plt.plot([i, i], [true_ate.iloc[0, 2], model_estimation.loc[0, 2]], color='black')
    # plt.title(f"{model_name} - ATE Estimations for T=2")
    plt.ylabel("ATE")
    plt.xlabel("Dataset")
    plt.xticks([])
    plt.scatter([], [], color='green', label='True ATE')
    plt.scatter([], [], color='purple', label='Model ATE')
    plt.legend(loc = 'upper right')
    save_path2 = os.path.join(save_path, f"{model_name}_T2.pdf")
    plt.savefig(save_path2)
    plt.cla()
    plt.clf()