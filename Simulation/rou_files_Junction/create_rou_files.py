import os
from xml.etree import ElementTree as ET
import copy
import pandas as pd
import matplotlib.pyplot as plt
from settings import *
import numpy as np

SIZE = 400


def create_vehicle_amounts(plot=False):
    size = SIZE
    grow_rates = [1, 1.2, 1.4, 1.6, 1.8, 2.0]
    decay_rates = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0]
    # every key stands for 10 min
    veh_amounts = {i: size * grow_rate for i, grow_rate in enumerate(grow_rates)}
    veh_amounts.update({i + 6: size * decay_rate for i, decay_rate in enumerate(decay_rates)})

    if plot:
        df = pd.DataFrame(veh_amounts.items(), columns=['10min', 'Vehicles'])
        df.plot(x='10min', y='Vehicles', kind='bar')
        plt.suptitle('Expected Vehicles demand per hour')
        plt.grid()
        plt.savefig(f'{ROOT}/rou_files_{EXP_NAME}/{EXP_NAME}.png')
        # plt.show()
        plt.close()
    return veh_amounts


def set_rou_file(seed, HOUR_LEN=600, plot=False):
    np.random.seed(seed)
    veh_amounts = create_vehicle_amounts(plot=plot)

    in_junctions = JUNCTIONS.copy()
    in_behaviors = {junc: (np.random.uniform(0.9, 1.1), np.random.uniform(0,0.2)) for junc in in_junctions}
    # round the in_behaviors
    out_junctions = JUNCTIONS.copy()

    # Load and parse the XML file
    tree = ET.parse(f'{ROOT}/rou_files_{EXP_NAME}/{EXP_NAME}.rou.xml')
    root = tree.getroot()


    for hour, hour_demand in veh_amounts.items():
        total_arrival_prob = hour_demand / 3600
        in_probs = np.random.uniform(0.8, 1.2, len(in_junctions))
        for in_junc in in_junctions:
            out_probs = np.random.uniform(0, 1, len(out_junctions) - 1)
            out_probs = out_probs / out_probs.sum()
            i = 0
            for out_junc in out_junctions:
                if in_junc == out_junc:
                    continue
                flow_prob = total_arrival_prob * out_probs[i] * in_probs[in_junctions.index(in_junc)]
                vType = ET.Element('vType', id=f"{hour}_{in_junc}_{out_junc}",
                                   speedFactor=f"normc({in_behaviors[in_junc][0]},{in_behaviors[in_junc][1]},0.2,2)")
                flow = ET.Element('flow', id=f'flow_{hour}_{in_junc}_{out_junc}', type=f"{hour}_{in_junc}_{out_junc}",
                                  begin=str(hour * HOUR_LEN),
                                  fromJunction=in_junc, toJunction=out_junc, end=str((hour + 1) * HOUR_LEN),
                                  probability=f"{flow_prob}", departSpeed="desired",)
                flow.tail = '\n\t'
                vType.tail = '\n\t'
                root.append(vType)
                root.append(flow)
                i += 1

    # Save the changes back to the file
    os.makedirs(f"{ROOT}/rou_files_{EXP_NAME}/{seed}", exist_ok=True)
    tree.write(f'{ROOT}/rou_files_{EXP_NAME}/{seed}/{EXP_NAME}.rou.xml')


if __name__ == '__main__':
    np.random.seed(SEED)
    seeds = [np.random.randint(0, 10000) for _ in range(NUM_EXPS)]
    for seed in seeds:
        set_rou_file(seed, plot=True)
