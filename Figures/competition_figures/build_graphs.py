import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FixedLocator
from Figures.competition_figures.ate_tables import build_basic_ate_table, build_paired_ate_table
from typing import List

dict_of_short_model_names = {"TMLE_Standardization_LinearRegression_IPW_GradientBoostingClassifier": "TMLE Std",
                             "PropensityMatching_GradientBoostingClassifier": "Propensity Matching",
                             "Matching_mahalanobis_5": "Matching Mal 5",
                             "Matching_mahalanobis_3": "Matching Mal 3",
                             "Matching_mahalanobis_1": "Matching Mal 1",
                             "Matching_euclidean_5": "Matching Euc 5",
                             "Matching_euclidean_3": "Matching Euc 3",
                             "Matching_euclidean_1": "Matching Euc 1",
                             "IPW_LogisticRegression(penalty='l1', solver='saga')": "IPW LR",
                             "Standardization_GradientBoostingRegressor": "Std GBR",
                             }

def build_bar_plots_from_dataframe(dataframe_ates: pd.DataFrame,
                                  title: str,
                                  save_path: str,
                                  x_label: str = None,
                                  y_label: str = None,
                                  dataframe_std: pd.DataFrame = None,
                                  reference_value: List[float] = None):
    """
    Creates 2 separate bar plot from a dataframe of ATEs estimations, one for T=1 and one for T=2. Saves them as pdfs.
    :param dataframe_ates: The dataframe, containing models performance against True ATEs for some error function.
    :param title: The title of the plot
    :param x_label: The x-axis label
    :param y_label: The y-axis label
    :param save_path: The path to save the plot
    :param dataframe_std: The dataframe, containing the standard deviation of the models performance against True ATEs
    :param reference_value: The True ATE value
    """
    models = dataframe_ates.index[:-1]  # Extract the models names
    for T in [1, 2]:
        fig, ax = plt.subplots(figsize=(10, 6))

        models_ates = {model_name: dataframe_ates.loc[model_name, f"T={T}"] for model_name in models} # Extract ATE values for the current T.

        if dataframe_std is not None:
            errors = dataframe_std.loc[T, models].values
            ax.bar([dict_of_short_model_names[model_name] for model_name in models_ates.keys()], models_ates.values(), yerr=errors, capsize=5, label=f"ATE of T={T} Estimates")
        else:
            ax.bar([dict_of_short_model_names[model_name] for model_name in models_ates.keys()], models_ates.values(), label=f"T={T} Estimates")

        if reference_value is not None:
            ax.axhline(y=reference_value[T-1], color='r', linestyle='--', label='True simulation value')  # Plot the true ATE

        if not y_label:
            ax.set_ylabel(y_label, fontsize=12)
        if not x_label:
            ax.set_xlabel(x_label, fontsize=12)

        ax.set_title(f"{title} (T={T})", fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend(fontsize=10)

        fig.tight_layout()
        plt.savefig(f"{save_path}_T={T}.pdf")
        plt.close(fig)

def build_basic_ate_plot(num_runs: int, save_path: str):
    """
    Create the ATE plot, each bar is a model's average ATE over all the datasets and there is a red line representing the
    true ATE.
    :param num_runs: The number of datasets
    :param save_path: The path to save the plot
    """
    ate_table = build_basic_ate_table(num_runs, ci=False)
    build_bar_plots_from_dataframe(ate_table.drop("True ATE"), "Average ATE of Each Model", save_path, y_label="Average of ATE estimations",
                                   reference_value=[ate_table.loc["True ATE", "T=1"], ate_table.loc["True ATE", "T=2"]])

def build_paired_ate_plot(num_runs: int, save_path: str, error_function: str = "relative_error", agg_function: str = "mean"):
    """
    Create the paired ATE plot, each bar is a model's average ATE over all the datasets and there is a red line representing the
    true ATE.
    :param num_runs: The number of datasets
    :param save_path: The path to save the plot
    :param error_function: The error function to use for calculating of the deviation from the true values
    :param agg_function: The aggregation function to use to aggragate the error values
    """
    ate_table = build_paired_ate_table(num_runs, error_function, agg_function)
    build_bar_plots_from_dataframe(ate_table, f"Average {error_function} of Each Model", save_path, x_label="Model", y_label=error_function)


if __name__ == '__main__':
    build_basic_ate_plot(2, "basic_ate_bar_graph")
    build_paired_ate_plot(2, "paired_ate_bar_graph_mean_re", error_function="relative_error", agg_function="mean")
