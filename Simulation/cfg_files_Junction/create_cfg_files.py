import os
from xml.etree import ElementTree as ET
import numpy as np
from settings import *


def set_cfg_file(type, seed):
    # Load and parse the XML file
    tree = ET.parse(f'{ROOT}/cfg_files_{EXP_NAME}/{EXP_NAME}_{type}.sumocfg')
    root = tree.getroot()

    # set route file
    route_file = root.find("input").find('route-files')
    route_file.set('value', f'{ROOT}/rou_files_{EXP_NAME}/{seed}/{EXP_NAME}.rou.xml')



    # set net file
    net_file = root.find("input").find('net-file')
    net_file.set('value', f'{ROOT}/cfg_files_{EXP_NAME}/{EXP_NAME}_{type}.net.xml')

    # set seed
    seed_element = root.find("random_number").find('seed')
    seed_element.set('value', str(seed))

    os.makedirs(f"{ROOT}/cfg_files_{EXP_NAME}/{type}/{seed}", exist_ok=True)
    # Save the changes back to the file
    tree.write(f'{ROOT}/cfg_files_{EXP_NAME}/{type}/{seed}/{EXP_NAME}.sumocfg')


if __name__ == "__main__":
    np.random.seed(SEED)

    seeds = [np.random.randint(0, 10000) for _ in range(NUM_EXPS)]
    for type in TYPES.keys():
        for seed in seeds:
            set_cfg_file(type, seed)