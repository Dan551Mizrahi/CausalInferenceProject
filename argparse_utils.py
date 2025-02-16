import argparse
import numpy as np
import time
import os

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description='Simulation and Policy Arguments')
    # Simulation arguments
    parser.add_argument("-s", '--seed', type=int, default=42,
                        help='Seed for the simulation')
    parser.add_argument("--num_experiments", type=int, default=500,
                        help='Number of experiments to run')
    parser.add_argument("-d", "--demand_size", type=int, default=180, help='Demand to run, None=all demands')
    parser.add_argument("--episode_len", type=int, default=600, help='Length of the episode')
    parser.add_argument("--lane_log_period", type=int, default=60, help='Period to log lane data')
    parser.add_argument("--gui", type=str2bool, default=False, help='Run with GUI')

    # Run arguments
    parser.add_argument("--run_simulation", type=str2bool, default=True, help='Run the simulation')
    parser.add_argument("--num_processes", type=int, default=None,
                        help='Number of processes to run in parallel, None=All available cores')
    parser.add_argument("--run_competition", type=str2bool, default=True, help='Run the competition')
    parser.add_argument("--parse_results", type=str2bool, default=True, help='Parse the results')
    parser.add_argument("--num_runs", type=int, default=50, help='Number of runs of the whole pipeline')
    parser.add_argument("--continue_from", type=int, default=50, help='Continue from run number')

    # Competition arguments
    parser.add_argument("-m", "--model", type=str, default=None, help="Model to run, None=all estimators")

    args = parser.parse_args()

    np.random.seed(args.seed)
    simulation_arguments = dict()

    # The total number of experiments. The results are divided to each run
    num_tot_experiment = args.num_experiments * args.num_runs
    num_skip_experiment = args.continue_from * args.num_experiments

    simulation_arguments["seed"] = list(np.random.choice(num_tot_experiment*100, num_skip_experiment+num_tot_experiment, replace=False)*2)
    simulation_arguments["seed"] = simulation_arguments["seed"][num_skip_experiment:]
    simulation_arguments["demand"] = [np.random.randint(args.demand_size * 2 // 3, args.demand_size * 4 // 3) for _ in
                                      range(num_tot_experiment)]
    simulation_arguments["episode_len"] = [args.episode_len for _ in range(num_tot_experiment)]
    simulation_arguments["lane_log_period"] = [args.lane_log_period for _ in range(num_tot_experiment)]
    simulation_arguments["gui"] = [args.gui for _ in range(num_tot_experiment)]
    simulation_arguments = [dict(zip(simulation_arguments.keys(), values)) for values in
                            zip(*simulation_arguments.values())]

    run_args = dict()
    run_args["num_processes"] = args.num_processes
    run_args["run_simulation"] = args.run_simulation
    run_args["run_competition"] = args.run_competition
    run_args["num_runs"] = args.num_runs
    run_args["continue_from"] = args.continue_from
    run_args["num_experiments"] = args.num_experiments
    run_args["parse_results"] = args.parse_results

    competition_args = dict()
    competition_args["model"] = args.model
    return run_args, simulation_arguments, competition_args

if __name__ == '__main__':
    args = get_args()