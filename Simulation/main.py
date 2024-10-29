from tqdm import tqdm
from multiprocessing import Pool
from run_utils.argparse_utils import get_args
from SUMO.SUMOAdapter import SUMOAdapter
from results_utils.results_parse import *
from run_utils.TL_policy import determine_policy

def simulate(simulation_arguments):
    # TODO: add results parse after every simulation run
    sumo = SUMOAdapter(**simulation_arguments)

    # run with T=0
    sumo.init_simulation()
    sumo.run_simulation()

    # determine policy
    tripinfo_output_file = sumo.tripinfo_file
    delay_sum = get_delay_sum(tripinfo_output_file)
    policy = determine_policy(delay_sum)

    # For true ATE calculation
    sumo.re_init_simulation(TL_type=1)
    sumo.run_simulation()
    sumo.re_init_simulation(TL_type=2)
    sumo.run_simulation()

    # run with determined policy
    new_seed = simulation_arguments["seed"] + 1
    sumo.re_init_simulation(seed=new_seed, TL_type=policy)
    sumo.run_simulation()

def main(simulation_arguments, run_args):
    with Pool(run_args["num_processes"]) as p:
        list(tqdm(p.imap(simulate, simulation_arguments), total=len(simulation_arguments)))

    if run_args["parse_results"]:
        calculate_ATEs(create_training_table())
        calculate_ATEs(create_testing_table(), training=False)

if __name__ == "__main__":
    main(*get_args())

