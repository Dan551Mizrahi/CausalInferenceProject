import os
import xml.etree.ElementTree as ET

demands = os.listdir()
for demand in demands:
    seeds = os.listdir(demand)
    seeds = [os.path.join(demand, seed) for seed in seeds]
    for seed in seeds:
        runs = os.listdir(seed)
        tripinfo_files = [os.path.join(seed,run) for run in runs if "tripinfo" in run]
        for tripinfo_file in tripinfo_files:
            try:
                tree = ET.parse(tripinfo_file)
                root = tree.getroot()
            except:
                print(f"Error in {tripinfo_file}")
                continue