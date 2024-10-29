import argparse
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description='Simulation and Policy Arguments')
    parser.add_argument("-s", '--seed', type=int, default=42,
                        help='Seed for the simulation')
    parser.add_argument("-n", "--num_experiments", type=int, default=1,
                        help='Number of experiments to run')
    parser.add_argument("--num_processes", type=int, default=None,
                        help='Number of processes to run in parallel, None=All available cores')
    parser.add_argument("-d", "--demand_size", type=int, default=400, help='Demand to run, None=all demands')
    parser.add_argument("--episode_len", type=int, default=600, help='Length of the episode')
    parser.add_argument("--lane_log_period", type=int, default=60, help='Period to log lane data')
    parser.add_argument("--parse_results", type=bool, default=False, help='Parse results')
    parser.add_argument("--gui", type=bool, default=False, help='Run with GUI')

    args = parser.parse_args()

    np.random.seed(args.seed)
    simulation_arguments = dict()
    simulation_arguments["seed"] = \
        [np.random.randint(1, 99999) for _ in range(args.num_experiments)]
    simulation_arguments["demand"] = [args.demand_size for _ in range(args.num_experiments)]
    simulation_arguments["episode_len"] = [args.episode_len for _ in range(args.num_experiments)]
    simulation_arguments["lane_log_period"] = [args.lane_log_period for _ in range(args.num_experiments)]
    simulation_arguments["gui"] = [args.gui for _ in range(args.num_experiments)]

    simulation_arguments = [dict(zip(simulation_arguments.keys(), values)) for values in zip(*simulation_arguments.values())]

    run_args = dict()
    run_args["num_processes"] = args.num_processes
    run_args["parse_results"] = args.parse_results

    return simulation_arguments, run_args
