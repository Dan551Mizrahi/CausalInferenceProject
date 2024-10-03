from settings import *
import xml.etree.ElementTree as ET
import pandas as pd

def output_file_to_df(output_file):
    # Parse the XML file into pd dataframe
    tree = ET.parse(output_file)
    root = tree.getroot()

    dict = {"departDelay": [], "timeLoss": [], "id": [], "depart":[], "speedFactor":[]}
    for tripinfo in root.findall('tripinfo'):
        for key in dict.keys():
            dict[key].append(tripinfo.get(key))
    df = pd.DataFrame(dict)
    df["totalDelay"] = df.departDelay.astype(float) + df.timeLoss.astype(float)
    df["departJunc"] = df["id"].apply(lambda x: x.split("_")[2])
    df["speedFactor"] = df["speedFactor"].astype(float)
    # convert to float except vType
    return df[["departJunc", "speedFactor", "totalDelay"]]

def create_training_table():
    total_dict = {"X":[],"B":[],"T":[],"Y":[]}
    for type, seeds in TRAINING_SEEDS.items():
        for seed in seeds:
            output_file = f"{RESULTS_FOLDER}/{type}/{seed}/{EXP_NAME}"
            df = output_file_to_df(output_file+ ".xml")
            with open(output_file+ "_snapshots.xml", "r") as f:
                snapshots = eval(f.read())
            mean_speedFactor = df.groupby("departJunc")["speedFactor"].mean().values
            dev_speedFactor = df.groupby("departJunc")["speedFactor"].std().values
            mean_delay = df["totalDelay"].mean()
            total_dict["X"].append(snapshots)
            total_dict["B"].append([(mSF, dSF) for mSF, dSF in zip(mean_speedFactor, dev_speedFactor)])
            total_dict["T"].append(type)
            total_dict["Y"].append(mean_delay)
    df = pd.DataFrame(total_dict)
    df.to_csv("Training_Table.csv")
    return df

def create_testing_table():
    total_dict = {"X":[],"B":[],"T":[],"Y":[]}
    for seed in TESTING_SEEDS:
        for type in TYPES.keys():
            output_file = f"{RESULTS_FOLDER}/{type}/{seed}/{EXP_NAME}"
            df = output_file_to_df(output_file+ ".xml")
            with open(output_file+ "_snapshots.xml", "r") as f:
                snapshots = eval(f.read())
            mean_speedFactor = df.groupby("departJunc")["speedFactor"].mean().values
            dev_speedFactor = df.groupby("departJunc")["speedFactor"].std().values
            mean_delay = df["totalDelay"].mean()
            total_dict["X"].append(snapshots)
            total_dict["B"].append([(mSF, dSF) for mSF, dSF in zip(mean_speedFactor, dev_speedFactor)])
            total_dict["T"].append(type)
            total_dict["Y"].append(mean_delay)
    df = pd.DataFrame(total_dict)
    df.to_csv("Testing_Table.csv")
    return df

def calculate_ATEs(df,training=True):
    T_means = df.groupby("T")["Y"].mean()
    ATEs = {t : [] for t in TYPES.keys()}
    for t in TYPES.keys():
        for t2 in TYPES.keys():
            ATEs[t].append(T_means[t] - T_means[t2])
    df_ATEs = pd.DataFrame(ATEs)
    if training:
        df_ATEs.to_csv("Training_ATEs.csv")
    else:
        df_ATEs.to_csv("Testing_ATEs.csv")
if __name__ == "__main__":
    calculate_ATEs(create_training_table())
    calculate_ATEs(create_testing_table(),training=False)