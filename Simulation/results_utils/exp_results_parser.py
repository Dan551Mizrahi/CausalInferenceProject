import pandas as pd
import xml.etree.ElementTree as ET
import warnings

warnings.filterwarnings("ignore")
def get_delay_sum(tripinfo_file):
    tree = ET.parse(tripinfo_file)
    root = tree.getroot()
    delay_sum = 0
    for tripinfo in root.findall('tripinfo'):
        delay_sum += float(tripinfo.get("departDelay")) + float(tripinfo.get("timeLoss"))
    return delay_sum

class ResultsParser:
    def __init__(self, tripinfo_file):
        self.tripinfo_file = tripinfo_file
        self.tripinfo_df = self._parse_tripinfo_output()



    def _parse_tripinfo_output(self):
        """
        Parse the XML file into pd dataframe`
        :param output_file: path to the output file (XML)
        :return: pd dataframe of the output file, with the following columns:
                    ['id', 'vType', 'numPass', 'duration, 'totalDelay', 'passDelay', 'passDuration']
        """
        tree = ET.parse(self.tripinfo_file)
        root = tree.getroot()

        results = {"departDelay": [], "timeLoss": [], "speedFactor": [], "departLane": []}
        for tripinfo in root.findall('tripinfo'):
            for key in results.keys():
                results[key].append(tripinfo.get(key))
        df = pd.DataFrame(results)
        df["totalDelay"] = df.departDelay.astype(float) + df.timeLoss.astype(float)
        df["departJunc"] = df["departLane"].apply(lambda x: x[0])
        df["speedFactor"] = df["speedFactor"].astype(float)
        return df[["departJunc", "speedFactor", "totalDelay"]]

    def get_b(self):
        grouped_juncs = self.tripinfo_df.groupby("departJunc")
        means = grouped_juncs["speedFactor"].mean()
        devs = grouped_juncs["speedFactor"].std()
        means_dict = means.to_dict()
        means_dict = {"b_mean_"+k: v for k, v in means_dict.items()}
        devs_dict = devs.to_dict()
        devs_dict = {"b_std_"+k: v for k, v in devs_dict.items()}
        means_dict.update(devs_dict)
        return means_dict

    def get_d(self):
        d_dict = self.tripinfo_df.groupby("departJunc").size().to_dict()
        d_dict = {"d_"+k: v for k, v in d_dict.items()}
        return d_dict

    def get_y(self):
        return {"Y": self.tripinfo_df["totalDelay"].sum()}



if __name__ == '__main__':
    tripinfo_file = "../SUMO/outputs/Test/0/0/Nothing_tripinfo.xml"
    lanes_file = "../SUMO/outputs/Test/0/0/Nothing_lanes.xml"
    parser = ResultsParser(tripinfo_file, lanes_file)
    print(parser.mean_speed_all_lanes())
