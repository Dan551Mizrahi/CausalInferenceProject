import argparse
import numpy as np
import time

def get_args():
    parser = argparse.ArgumentParser(description='Simulation and Policy Arguments')
    parser.add_argument("-s", '--seed', type=int, default=42,
                        help='Seed for the simulation')
    parser.add_argument("-n", "--num_experiments", type=int, default=100,
                        help='Number of experiments to run')
    parser.add_argument("--num_processes", type=int, default=None,
                        help='Number of processes to run in parallel, None=All available cores')
    parser.add_argument("-d", "--demand_size", type=int, default=180, help='Demand to run, None=all demands')
    parser.add_argument("--episode_len", type=int, default=600, help='Length of the episode')
    parser.add_argument("--lane_log_period", type=int, default=60, help='Period to log lane data')
    parser.add_argument("--gui", type=bool, default=False, help='Run with GUI')

    args = parser.parse_args()

    np.random.seed(args.seed)
    simulation_arguments = dict()
    simulation_arguments["seed"] = \
        [np.random.randint(1, 99999) for _ in range(args.num_experiments)]
    simulation_arguments["demand"] = [np.random.randint(args.demand_size*2//3,args.demand_size*4//3) for _ in range(args.num_experiments)]
    simulation_arguments["episode_len"] = [args.episode_len for _ in range(args.num_experiments)]
    simulation_arguments["lane_log_period"] = [args.lane_log_period for _ in range(args.num_experiments)]
    simulation_arguments["gui"] = [args.gui for _ in range(args.num_experiments)]

    simulation_arguments = [dict(zip(simulation_arguments.keys(), values)) for values in zip(*simulation_arguments.values())]

    # write simulation arguments to file
    with open(f"run_logs/{time.time()}_simulation_args.txt", "w") as f:
        f.writelines([f"{key}: {value}\n" for key, value in simulation_arguments[0].items()])

    run_args = dict()
    run_args["num_processes"] = args.num_processes

    return simulation_arguments, run_args
