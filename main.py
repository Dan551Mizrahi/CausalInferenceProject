import Simulation.run_simulation as run_simulation
from ATE_calculator.bootstrap_ATE import *
from causal_inference_models.models import *

def main():
    # training_df, testing_df = run_simulation.main()
    # training_df.to_csv("Simulated_Data/Training_data.csv")
    # testing_df.to_csv("Simulated_Data/Testing_data.csv")

    # training_df = pd.read_csv("Simulated_Data/Training_data.csv")
    testing_df = pd.read_csv("Simulated_Data/Testing_data.csv")
    testing_ATEs = calculate_ATEs(testing_df)
    testing_ATEs.to_csv("Simulated_Data/testing_ATEs.csv")

    # testing_ATEs_with_CI = bootstrap_ATEs(testing_df)
    # testing_ATEs_with_CI.to_csv("Simulated_Data/Testing_ATEs_with_CI.csv")
    # training_ATEs = calculate_ATEs(training_df)
    # training_ATEs.to_csv("Simulated_Data/Training_ATEs.csv")

    df = pd.read_csv("/Users/danmizrahi/Desktop/causalInference/CausalInferenceProject/Simulated_Data/Training_data_3000.csv", index_col=0)
    df = pd.get_dummies(df, dtype=float)

    cols_to_scale = ['b_mean_E', 'b_mean_N', 'b_mean_S', 'b_mean_W', 'b_std_E',
                     'b_std_N', 'b_std_S', 'b_std_W', 'd_E', 'd_N', 'd_S']
    scale = MinMaxScaler()
    df[cols_to_scale] = scale.fit_transform(df[cols_to_scale])

    data = DataCIModel(df)

    causal_inference_estimations = CausalInferenceEstimations()

    ATE_dict = causal_inference_estimations.estimate_with_all_models(data)

    # Get true ATEs
    true_ATEs = pd.read_csv("/Users/danmizrahi/Desktop/causalInference/CausalInferenceProject/Simulated_Data/testing_ATEs.csv", index_col=0)
    # Get test data
    # Test whole data
    test_data = pd.read_csv(
        "/Users/danmizrahi/Desktop/causalInference/CausalInferenceProject/Simulated_Data/Testing_data_3000.csv",
        index_col=0)

    for idx in [[0,1], [0,2], [1,2]]:
         causal_inference_estimations.graph_ATE(ATE_dict, idx, true_ATEs.iloc[idx[0], idx[1]], data_set_name="Train_3000")

    l1_re_graphs(causal_inference_estimations, data, test_data, "Train_3000")

    lr = LogisticRegression(max_iter=10000, random_state=42)
    lr.fit(data.X, data.T)
    df['propensity_score_lr'] = lr.predict_proba(data.X)[:, 1]
    filtered_df = df.loc[df['propensity_score_lr'] > 0.1]
    filtered_df = filtered_df.loc[filtered_df['propensity_score_lr'] < 0.4]
    data = DataCIModel(filtered_df)
    causal_inference_estimations = CausalInferenceEstimations()
    ATE_dict = causal_inference_estimations.estimate_with_all_models(data)

    for idx in [[0,1], [0,2], [1,2]]:
            causal_inference_estimations.graph_ATE(ATE_dict, idx, true_ATEs.iloc[idx[0], idx[1]], data_set_name="Train_3000_filtered_by_overlap")

    l1_re_graphs(causal_inference_estimations, data, test_data, "Train_3000_filtered_by_overlap")
    df = pd.read_csv("/Users/danmizrahi/Desktop/causalInference/CausalInferenceProject/Simulated_Data/Training_data.csv",
        index_col=0)
    df = pd.get_dummies(df, dtype=float)

    cols_to_scale = ['b_mean_E', 'b_mean_N', 'b_mean_S', 'b_mean_W', 'b_std_E',
                     'b_std_N', 'b_std_S', 'b_std_W', 'd_E', 'd_N', 'd_S']
    scale = MinMaxScaler()
    df[cols_to_scale] = scale.fit_transform(df[cols_to_scale])

    data = DataCIModel(df)

    causal_inference_estimations = CausalInferenceEstimations()

    ATE_dict = causal_inference_estimations.estimate_with_all_models(data)

    for idx in [[0, 1], [0, 2], [1, 2]]:
        causal_inference_estimations.graph_ATE(ATE_dict, idx, true_ATEs.iloc[idx[0], idx[1]],
                                               data_set_name="Train_100")

    l1_re_graphs(causal_inference_estimations, data, test_data, "Train_100")


if __name__ == '__main__':
    main()
