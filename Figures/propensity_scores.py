from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(current_dir)

def propensity_graphs(df: pd.DataFrame, save_path: str):
    """
    Uses the Gradient Boosting Classifier to estimate the propensity scores.
    Evaluate the propensity scores by the model's AUC and plot the propensity score overlap.
    :param df: The data frame
    :param save_path: The path to save the graphs
    """
    X = df.drop(columns=['T', 'Y'])
    a = df['T']

    X_train, X_test, a_train, a_test = train_test_split(X, a, test_size=0.2, random_state=42)

    # Initialize the Gradient Boosting Classifier
    gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

    gbm.fit(X_train, a_train)

    propensity_scores = gbm.predict_proba(X_test)

    auc = roc_auc_score(a_test, propensity_scores, multi_class='ovo')
    print(f"AUC ovo: {auc}")
    auc = roc_auc_score(a_test, propensity_scores, multi_class='ovr')
    print(f"AUC ovr: {auc}")

    df['propensity_score_gbm'] = gbm.predict_proba(X)[:, 1]

    sns.histplot(df[df['T'] == 2]['propensity_score_gbm'], label='Smart TL (T=2)', kde=True)
    sns.histplot(df[df['T'] == 1]['propensity_score_gbm'], label='Traffic Light (T=1)', kde=True)
    sns.histplot(df[df['T'] == 0]['propensity_score_gbm'], label='No TL (T=0)', kde=True)
    plt.xlabel('Propensity Score')
    plt.ylabel('Density')
    plt.title('Propensity Score Overlap')
    plt.legend()
    plt.show()

    filtered_df = df.loc[df['propensity_score_gbm'] > 0.1]
    filtered_df = filtered_df.loc[filtered_df['propensity_score_gbm'] < 0.4]
    sns.histplot(filtered_df[filtered_df['T'] == 2]['propensity_score_gbm'], label='Smart TL (T=2)', kde=True)
    sns.histplot(filtered_df[filtered_df['T'] == 1]['propensity_score_gbm'], label='Traffic Light (T=1)', kde=True)
    sns.histplot(filtered_df[filtered_df['T'] == 0]['propensity_score_gbm'], label='No TL (T=0)', kde=True)
    plt.xlabel('Propensity Score')
    plt.ylabel('Density')
    plt.title('Propensity Score Overlap')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    df = pd.read_pickle(os.path.join(project_dir, "Simulation", "simulated_data", "run_5", "training_data.pkl"))
    propensity_graphs(df, project_dir)

    df = pd.read_pickle(os.path.join(project_dir, "Simulation", "simulated_data", "run_7", "training_data.pkl"))
    propensity_graphs(df, project_dir)