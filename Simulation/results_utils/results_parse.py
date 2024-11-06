import xml.etree.ElementTree as ET
import pandas as pd


def create_row(rp, prior, policy):
    row = {"Prior": prior}
    row.update(rp.get_b())
    row.update(rp.get_d())
    row.update({"T": policy})
    row.update(rp.get_y())
    return row


def calculate_ATEs(df, training=True):
    T_means = df.groupby("T")["Y"].mean()
    treatments = sorted(df["T"].unique().tolist())
    ATEs = {t: [] for t in treatments}
    for t in treatments:
        for t2 in treatments:
            ATEs[t].append(T_means[t] - T_means[t2])
    df_ATEs = pd.DataFrame(ATEs)
    if training:
        df_ATEs.to_csv("Training_ATEs.csv")
    else:
        df_ATEs.to_csv("Testing_ATEs.csv")
