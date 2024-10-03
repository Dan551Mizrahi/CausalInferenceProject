import numpy as np
# Paths
ROOT = r"C:\CI\CausalInferenceProject\Simulation"
RESULTS_FOLDER = f"{ROOT}/results_reps"
# Experiment Settings
EXP_NAME = "Junction"
JUNCTIONS = ["W","E","S","N"]
EDGES = [f"C_{junc}" for junc in JUNCTIONS] + [f"{junc}_C" for junc in JUNCTIONS]
TYPES = {0:"PRIORITY",1:"STATIC",2:"DELAY", 3:"ACTUATED"}
# Machine Settings
GUI = False
NUM_PROCESSES = 10

# Randomization Settings
SEED = 42
NUM_EXPS = 100


np.random.seed(SEED)
ALL_SEEDS = np.random.randint(0, 10000,NUM_EXPS)
TRAINING_SEEDS = {key: ALL_SEEDS[key*NUM_EXPS//5:(key+1)*NUM_EXPS//5] for key in TYPES.keys()}
TESTING_SEEDS = ALL_SEEDS[-NUM_EXPS//5:]