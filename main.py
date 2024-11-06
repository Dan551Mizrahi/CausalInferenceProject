import Simulation.run_simulation as run_simulation


def main():
    training_df, testing_df = run_simulation.main()
    training_df.to_csv("Simulated_Data/Training_data.csv")
    testing_df.to_csv("Simulated_Data/Testing_data.csv")

    calculate_ATEs(training_df)
    calculate_ATEs(testing_df, training=False)


if __name__ == '__main__':
    main()
