import xml.etree.ElementTree as ET
import pandas as pd


def create_row(rp, prior, policy):
    row = {"Prior": prior}
    row.update(rp.get_b())
    row.update(rp.get_d())
    row.update({"T": policy})
    row.update(rp.get_y())
    return row


