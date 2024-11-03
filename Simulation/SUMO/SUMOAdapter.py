import sys

import sumolib
import traci
import os
import numpy as np
import xml.etree.ElementTree as ET

EXP_NAME = "Junction"


class SUMOAdapter:
    def __init__(self,
                 seed: int = 42,
                 demand: int = 400,
                 episode_len: int = 600,
                 lane_log_period: int = 60,
                 gui: bool = False,
                 ):
        # validate inputs
        assert demand > 0, "demand must be a positive integer"
        assert episode_len > 0, "episode_len must be a positive integer"

        # simulation inputs
        self.seed = seed
        self.veh_amount = demand
        self.episode_len = episode_len
        self.lane_log_period = lane_log_period
        self.TL_type = 0

        # templates data
        curdir = os.path.dirname(__file__)
        self.template_files_folder = os.path.join(curdir, "template_files")
        self.net_file = os.path.join(self.template_files_folder, f"{EXP_NAME}_{self.TL_type}.net.xml")
        self.rou_file_template = os.path.join(self.template_files_folder, f"{EXP_NAME}.rou.xml")
        self.sumo_cfg_template = os.path.join(self.template_files_folder, f"{EXP_NAME}.sumocfg")
        self.add_file_template = os.path.join(self.template_files_folder, f"{EXP_NAME}.add.xml")

        # simulation files - will be created on init
        self.cfg_folder = os.path.join(curdir, "cfg_files", f"demand_{demand}", f"seed_{seed}")
        os.makedirs(self.cfg_folder, exist_ok=True)
        self.rou_file = None
        self.sumo_cfg = None
        self.add_file = None

        # output files - will be created after simulation
        self.output_folder = os.path.join(curdir, "outputs", f"demand_{demand}", f"seed_{seed}")
        os.makedirs(self.output_folder, exist_ok=True)
        self.output_name = f"{EXP_NAME}_{self.TL_type}_{self.seed}"
        self.tripinfo_file = None
        self.lanes_file = None

        # experiment constants
        self.junctions = ["W", "E", "S", "N"]
        self.gui = gui

    def init_simulation(self):
        self._set_rou_file()
        self._set_add_file()
        self._set_sumo_cfg()
        self._start_sumo_simulation()

    def re_init_simulation(self, seed: int = None, TL_type: int = None, chosen = False):
        # The input seed here is the realization of the arrival process and behavior of the vehicles
        assert seed is not None or TL_type is not None, "seed or TL_type must be set"
        # TODO: DO NOT SET ROU FILE AGAIN
        self.seed = seed if seed is not None else self.seed
        self.TL_type = TL_type if TL_type is not None else self.TL_type
        self.output_name = f"{EXP_NAME}_{self.TL_type}_{self.seed}"
        if chosen:
            self.output_name += "_chosen"
        self._set_add_file()
        self.net_file = os.path.join(self.template_files_folder, f"{EXP_NAME}_{TL_type}.net.xml")
        self._set_sumo_cfg()
        self._start_sumo_simulation()

    def run_simulation(self):
        while not self.isDone():
            self.step()
        self.close()

    def close(self):
        traci.close()

    def step(self):
        traci.simulationStep()

    def isDone(self):
        return traci.simulation.getMinExpectedNumber() <= 0

    def _create_vehicle_amounts(self):
        size = self.veh_amount
        grow_rates = [1, 1.2, 1.4, 1.6, 1.8, 2.0]
        decay_rates = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0]
        # every key stands for 10 min
        veh_amounts = {i: size * grow_rate for i, grow_rate in enumerate(grow_rates)}
        veh_amounts.update({i + 6: size * decay_rate for i, decay_rate in enumerate(decay_rates)})
        return veh_amounts

    def _set_rou_file(self):
        np.random.seed(self.seed)
        veh_amounts = self._create_vehicle_amounts()

        in_junctions = self.junctions
        in_behaviors = {junc: (np.random.uniform(0.9, 1.1), np.random.uniform(0, 0.2)) for junc in in_junctions}
        # round the in_behaviors
        out_junctions = self.junctions

        # Load and parse the XML file
        tree = ET.parse(self.rou_file_template)
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
                    flow = ET.Element('flow', id=f'flow_{hour}_{in_junc}_{out_junc}',
                                      type=f"{hour}_{in_junc}_{out_junc}",
                                      begin=str(hour * self.episode_len),
                                      fromJunction=in_junc, toJunction=out_junc, end=str((hour + 1) * self.episode_len),
                                      probability=f"{flow_prob}", departSpeed="max")
                    flow.tail = '\n\t'
                    vType.tail = '\n\t'
                    root.append(vType)
                    root.append(flow)
                    i += 1

        # Save the changes back to the file
        self.rou_file = os.path.join(self.cfg_folder, f"{EXP_NAME}.rou.xml")
        tree.write(self.rou_file)

    def _set_add_file(self):
        # Load and parse the XML file
        tree = ET.parse(self.add_file_template)
        root = tree.getroot()

        self.lanes_file = os.path.join(self.output_folder, f"{self.output_name}_lanes.xml")
        elem = ET.Element("laneData", id="lane_data", freq=str(self.lane_log_period), file=self.lanes_file)
        elem.tail = '\n\t'
        root.append(elem)

        # Save the changes back to the file
        self.add_file = os.path.join(self.cfg_folder, f"{self.output_name}.add.xml")
        tree.write(self.add_file)

    def _set_sumo_cfg(self):
        assert self.rou_file is not None, "rou file must be set before setting sumo cfg"
        assert self.net_file is not None, "net file must be set before setting sumo cfg"
        assert self.add_file is not None, "add file must be set before setting sumo cfg"
        # Load and parse the XML file
        tree = ET.parse(self.sumo_cfg_template)
        root = tree.getroot()

        # set route file
        route_file = root.find("input").find('route-files')
        route_file.set('value', self.rou_file)

        # set net file
        net_file = root.find("input").find('net-file')
        net_file.set('value', self.net_file)

        # set add file
        add_file = root.find("input").find('additional-files')
        add_file.set('value', self.add_file)

        # set seed
        seed_element = root.find("random_number").find('seed')
        seed_element.set('value', str(self.seed))

        # set tripinfo output file
        self.tripinfo_file = os.path.join(self.output_folder, f"{self.output_name}_tripinfo.xml")
        tripinfo = root.find("output").find('tripinfo-output')
        tripinfo.set('value', self.tripinfo_file)

        # Save the changes back to the file
        self.sumo_cfg = os.path.join(self.cfg_folder, f"{self.output_name}.sumocfg")
        tree.write(self.sumo_cfg)

    def _start_sumo_simulation(self):
        if 'SUMO_HOME' in os.environ:
            sumo_path = os.environ['SUMO_HOME']
            sys.path.append(os.path.join(sumo_path, 'tools'))
            # check operational system - if it is windows, use sumo.exe if linux, use sumo
            if os.name == 'nt':
                sumoBinary = os.path.join(sumo_path, 'bin', 'sumo-gui.exe') if self.gui else \
                    os.path.join(sumo_path, 'bin', 'sumo.exe')
            else:
                sumoBinary = os.path.join(sumo_path, 'bin', 'sumo-gui') if self.gui else \
                    os.path.join(sumo_path, 'bin', 'sumo')
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        sumoCmd = [sumoBinary, "-c", self.sumo_cfg]
        traci.start(sumoCmd)
