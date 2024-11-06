from tqdm import tqdm
from multiprocessing import Pool
from Simulation.run_utils.argparse_utils import get_args
from Simulation.SUMO.SUMOAdapter import SUMOAdapter
from Simulation.results_utils.results_parse import *
from Simulation.run_utils.TL_policy import determine_policy
from Simulation.results_utils.exp_results_parser import *


def simulate(simulation_arguments):
    sumo = SUMOAdapter(**simulation_arguments)

    # run with T=0
    sumo.init_simulation()
    sumo.run_simulation()

    # determine policy
    delay_sum = get_delay_sum(sumo.tripinfo_file)
    policy = determine_policy(delay_sum)

    # run with determined policy
    new_seed = simulation_arguments["seed"] + 1
    sumo.re_init_simulation(seed=new_seed, TL_type=policy, chosen=True)
    sumo.run_simulation()
    training_row = create_row(ResultsParser(sumo.tripinfo_file), delay_sum, policy)

    testing_rows = []
    # For true ATE calculation
    sumo.re_init_simulation(TL_type=0)
    sumo.run_simulation()
    testing_rows.append(create_row(ResultsParser(sumo.tripinfo_file), delay_sum, 0))
    sumo.re_init_simulation(TL_type=1)
    sumo.run_simulation()
    testing_rows.append(create_row(ResultsParser(sumo.tripinfo_file), delay_sum, 1))
    sumo.re_init_simulation(TL_type=2)
    sumo.run_simulation()
    testing_rows.append(create_row(ResultsParser(sumo.tripinfo_file), delay_sum, 2))

    return training_row, testing_rows


def simulate_all(simulation_arguments, run_args):
    with Pool(run_args["num_processes"]) as p:
        results = list(tqdm(p.imap(simulate, simulation_arguments), total=len(simulation_arguments)))

    training_table, testing_table = [], []
    for training_row, testing_rows in results:
        training_table.append(training_row)
        testing_table += testing_rows

    training_df = pd.DataFrame(training_table)
    testing_df = pd.DataFrame(testing_table)

    return training_df,testing_df


def main():
    simulate_all(*get_args())


if __name__ == "__main__":
    main()
