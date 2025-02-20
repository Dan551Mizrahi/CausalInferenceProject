# Creating Figures for CI Competition Analysis

This folder contains methods for analysing the performance of causal inference methods and creating figures on the generated data.

## Directory Contents
<ul>
<li><code>/</code>
<ul>
<li><code>run_figures.py</code> - A script that creates the graphs we used for our paper.</li>
<li><code>results_reading.py</code> - A script that this folder uses to read the ATE estimations of the models and the true results. No function is reading the results itself.</li>
<li><code>ate_tables_and_plots.py</code> - Creates tables and plots analysing the results with different measures (can be customized by changing the corresponding parameters).</li>
<li><code>propensity_scores.py</code> - This codes help to understand a dataset of your choosing, its propensity scores and its degree of overlap.</li>
<li><code>error_functions.py</code> - Here there are some metrics which are used as cost function when constructing the graphs and tables. You can add and modify them and then use them as parameters to a different function in this folder.</li>
<li><code>agg_functions.py</code> - Here there are some aggregation functions, e.g. mean, median, ... You can add and modify them and then use them as parameters to a different function in this folder. </li>
</ul>
</li>
</ul>

## Output
The graphs, tables and different outputs will be saved in a new folder named <code>plots_and_tables</code> created when calling the function <code>main_figures</code> inside <code>run_figures.py</code>.