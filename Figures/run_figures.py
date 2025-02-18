from Figures.ate_tables_and_plots import export_ate_table_excel, build_box_plots_graph, create_paired_graph
from Figures.results_reading import read_all_true_ates, read_all_model_estimates
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

def check_normality(num_runs: int, save_path: str):
    true_ates = read_all_true_ates(num_runs)
    list_of_t1_ates = []
    list_of_t2_ates = []
    for table in true_ates:
        dataframe = pd.DataFrame(table)
        list_of_t1_ates.append(dataframe.loc['T=0', 'T=1'])
        list_of_t2_ates.append(dataframe.loc['T=0', 'T=2'])

    # Draw the distribution of the True ATEs
    plt.hist(list_of_t1_ates, bins=20, alpha=0.5, label='ATEs T=1')
    plt.hist(list_of_t2_ates, bins=20, alpha=0.5, label='ATEs T=2')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_path, "True_ATEs_distributions.png"))
    plt.cla()
    plt.clf()

    # QQ plot for T=1
    stats.probplot(list_of_t1_ates, dist="norm", plot=plt)
    plt.savefig(os.path.join(save_path, "QQ_true_T1.png"))
    plt.cla()
    plt.clf()

    # QQ plot for T=2
    stats.probplot(list_of_t2_ates, dist="norm", plot=plt)
    plt.savefig(os.path.join(save_path, "QQ_true_T2.png"))
    plt.cla()
    plt.clf()

def t_test(num_runs: int, model_name: str, save_path: str):
    true_ates = read_all_true_ates(num_runs)
    list_of_t1_ates = []
    list_of_t2_ates = []
    for table in true_ates:
        dataframe = pd.DataFrame(table)
        list_of_t1_ates.append(dataframe.loc['T=0', 'T=1'])
        list_of_t2_ates.append(dataframe.loc['T=0', 'T=2'])

    # Read model estimations
    model_estimations = read_all_model_estimates(model_name, num_runs)
    model_estimations = model_estimations[model_name]
    list_of_t1_estimates = []
    list_of_t2_estimates = []
    for i in range(len(true_ates)):
        list_of_t1_estimates.append(model_estimations[i].loc[0, 1])
        list_of_t2_estimates.append(model_estimations[i].loc[0, 2])

    # T-test for T=1
    t_statistic, p_value = stats.ttest_rel(list_of_t1_ates, list_of_t1_estimates)
    with open(os.path.join(save_path, "t_test.txt"), "a") as f:
        f.write(f"T-test for T=1: t_statistic={t_statistic}, p_value={p_value}\n")

    # T-test for T=2
    t_statistic, p_value = stats.ttest_rel(list_of_t2_ates, list_of_t2_estimates)
    with open(os.path.join(save_path, "t_test.txt"), "a") as f:
        f.write(f"T-test for T=2: t_statistic={t_statistic}, p_value={p_value}\n")



def main_figures(num_runs: int, *args, **kwargs):
    # Create a directory for all the figures
    current_dir = os.path.dirname(__file__)
    if not os.path.exists(os.path.join(current_dir, "plots_and_tables")):
        os.makedirs(os.path.join(current_dir, "plots_and_tables"))
    current_inner_dir = os.path.dirname(__file__)

    export_ate_table_excel(num_runs, os.path.join(current_inner_dir, "plots_and_tables/basic_ate_table.xlsx"))
    export_ate_table_excel(num_runs, os.path.join(current_inner_dir, "plots_and_tables/relative_error_ate_table.xlsx"), type="paired",
                           error_function="relative_error", ci=True)
    build_box_plots_graph(num_runs, os.path.join(current_inner_dir, "plots_and_tables/"), trim_y_axis=True)
    create_paired_graph(num_runs, "PropensityMatching_GradientBoostingClassifier", os.path.join(current_inner_dir, "plots_and_tables/"))
    create_paired_graph(num_runs, "IPW_LogisticRegression(penalty='l1', solver='saga')", os.path.join(current_inner_dir, "plots_and_tables/"))
    check_normality(num_runs, os.path.join(current_inner_dir, "plots_and_tables/"))
    t_test(num_runs, "PropensityMatching_GradientBoostingClassifier", os.path.join(current_inner_dir, "plots_and_tables/"))
    t_test(num_runs, "IPW_LogisticRegression(penalty='l1', solver='saga')", os.path.join(current_inner_dir, "plots_and_tables/"))

if __name__ == '__main__':
    main_figures(50)