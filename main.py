import Simulation.run_simulation as run_simulation
from ATE_calculator.bootstrap_ATE import *

def main():
    # training_df, testing_df = run_simulation.main()
    # training_df.to_csv("Simulated_Data/Training_data.csv")
    # testing_df.to_csv("Simulated_Data/Testing_data.csv")

    training_df = pd.read_csv("Simulated_Data/Training_data.csv")
    testing_df = pd.read_csv("Simulated_Data/Testing_data.csv")

    testing_ATEs_with_CI = bootstrap_ATEs(testing_df)
    testing_ATEs_with_CI.to_csv("Simulated_Data/Testing_ATEs_with_CI.csv")
    training_ATEs = calculate_ATEs(training_df)
    training_ATEs.to_csv("Simulated_Data/Training_ATEs.csv")


if __name__ == '__main__':
    main()
