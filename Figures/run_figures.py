from Figures.ate_tables_and_plots import export_ate_table_excel, build_box_plots_graph, create_paired_graph
import os


def main_figures(num_runs: int, *args, **kwargs):
    # Create a directory for all the figures
    current_dir = os.path.dirname(__file__)
    if not os.path.exists(os.path.join(current_dir, "plots_and_tables")):
        os.makedirs(os.path.join(current_dir, "plots_and_tables"))
    current_inner_dir = os.path.dirname(__file__)

    export_ate_table_excel(50, os.path.join(current_inner_dir, "plots_and_tables/basic_ate_table.xlsx"))
    export_ate_table_excel(50, os.path.join(current_inner_dir, "plots_and_tables/relative_error_ate_table.xlsx"), type="paired",
                           error_function="relative_error", ci=True)
    build_box_plots_graph(50, os.path.join(current_inner_dir, "plots_and_tables/"), trim_y_axis=True)
    create_paired_graph(50, "PropensityMatching_GradientBoostingClassifier", os.path.join(current_inner_dir, "plots_and_tables/"))
    create_paired_graph(50, "IPW_LogisticRegression(penalty='l1', solver='saga')", os.path.join(current_inner_dir, "plots_and_tables/"))

if __name__ == '__main__':
    main_figures(50)