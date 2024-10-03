import os
import sys

from tqdm import tqdm
from multiprocessing import Pool

from settings import *
import traci
from results_parse import *

SNAPSHOT_RATE = 60

if GUI:
    NUM_PROCESSES = 1

if 'SUMO_HOME' in os.environ:
    sumo_path = os.environ['SUMO_HOME']
    sys.path.append(os.path.join(sumo_path, 'tools'))
    # check operational system - if it is windows, use sumo.exe if linux, use sumo
    if os.name == 'nt':
        sumoBinary = os.path.join(sumo_path, 'bin', 'sumo-gui.exe') if GUI else \
            os.path.join(sumo_path, 'bin', 'sumo.exe')
    else:
        sumoBinary = os.path.join(sumo_path, 'bin', 'sumo-gui') if GUI else \
            os.path.join(sumo_path, 'bin', 'sumo')
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


def init_simulation(sumoCfg):
    seed = sumoCfg.split("/")[-2]
    type = sumoCfg.split("/")[-3]
    sumoCmd = [sumoBinary, "-c", sumoCfg, "--tripinfo-output"]
    exp_output_name = f"{RESULTS_FOLDER}/{type}/{seed}/"
    os.makedirs(exp_output_name, exist_ok=True)
    exp_output_name += EXP_NAME + ".xml"
    sumoCmd.append(exp_output_name)
    traci.start(sumoCmd)
    return exp_output_name


def snapshot_features():
    snapshot_data = []
    for direction in JUNCTIONS:
        snapshot_data.append(traci.edge.getLastStepVehicleNumber(f"{direction}_C"))
    return snapshot_data

def simulate(sumoCfgPath):
    exp_output = init_simulation(sumoCfgPath)
    exp_output = ".".join(exp_output.split(".")[:-1]) + "_snapshots.xml"
    step = 0
    snapshot_data = []
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep(step)
        step += 1
        if step % SNAPSHOT_RATE == 0:
            snapshot_data.append(snapshot_features())
    with open(exp_output, "w") as f:
        f.write(str(snapshot_data))

    traci.close()
    # wandb.finish()


def parallel_simulation(sumoCfgPaths):
    with Pool(NUM_PROCESSES) as pool:
        results = list(tqdm(pool.imap(simulate, sumoCfgPaths), total=len(sumoCfgPaths)))




def run_random_experiments():
    sumoCfgPaths = []
    for type in TYPES.keys():
        for seed in ALL_SEEDS:
            sumoCfgPaths.append(f"cfg_files_{EXP_NAME}/{type}/{seed}/{EXP_NAME}.sumocfg")
    print("Number of sumoCfg files: ", len(sumoCfgPaths))
    if GUI:
        sumoCfgPaths = [sumoCfgPaths[3]]
    parallel_simulation(sumoCfgPaths)


def run_experiment(type, seed):
    sumoCfgPaths = [f"cfg_files_{EXP_NAME}/{type}/{seed}/{EXP_NAME}.sumocfg"]
    parallel_simulation(sumoCfgPaths)


def main():
    run_random_experiments()

if __name__ == "__main__":
    run_random_experiments()
    calculate_ATEs(create_training_table())
    calculate_ATEs(create_testing_table(), training=False)

