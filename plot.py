import sys
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataset_io import format_large_number

def read_stats_file(fname):

    row = {}
    stats = json.load(open(fname, "r"))

    parameters = stats["parameters"]
    row["infile"] = parameters["infile"].split("/")[-1]
    row["method"] = parameters["method"]
    row["cover"] = parameters["cover"]
    row["leaf_size"] = parameters["leaf_size"]
    row["num_centers"] = parameters["num_centers"]
    row["tree_assignment"] = parameters["tree_assignment"]
    row["query_balancing"] = parameters["query_balancing"]

    row["runtime"] = stats["runtime"]
    row["num_points"] = stats["num_points"]
    row["num_edges"] = stats["num_edges"]
    row["num_procs"] = stats["num_procs"]
    row["dist_comps"] = format_large_number(sum(stats["dist_comps"]))
    return row

def steal_stats_table(fname):

    stats = json.load(open(fname, "r"))

    assert stats["parameters"]["method"] == "gvor" and stats["parameters"]["query_balancing"] == "steal"

    num_procs = stats["num_procs"]
    rows = []

    for i in range(num_procs):
        row = {}
        row["rank"] = i
        row["comp_time"] = stats["my_steal_comp_time"][i]
        row["steal_time"] = stats["my_steal_time"][i]
        row["poll_time"] = stats["my_poll_time"][i]
        row["response_time"] = stats["my_response_time"][i]
        row["allreduce_time"] = stats["my_allreduce_time"][i]
        row["dist_comps"] = format_large_number(stats["dist_comps"][i])
        row["steal_successes"] = stats["steal_successes"][i]
        row["steal_attempts"] = stats["steal_attempts"][i]
        row["steal_services"] = stats["steal_services"][i]
        rows.append(row)

    return num_procs, pd.DataFrame(rows, columns=["rank", "comp_time", "steal_time", "poll_time", "response_time", "allreduce_time", "steal_successes", "steal_attempts", "steal_services", "dist_comps"])

rows = []

for path in Path("blob_results").glob("*.json"):
    fname = str(path)
    rows.append(read_stats_file(fname))

blob_table = pd.DataFrame(rows, columns=["infile", "method", "num_procs", "runtime", "dist_comps", "num_centers", "query_balancing", "num_points", "num_edges"])
blob_table = blob_table.sort_values(by=["num_procs", "method", "query_balancing"])
blob_table.round(3).to_csv("blob_results/table.csv")

rows = []

for path in Path("normal_results").glob("*.json"):
    fname = str(path)
    rows.append(read_stats_file(fname))

normal_table = pd.DataFrame(rows, columns=["infile", "method", "num_procs", "runtime", "dist_comps", "num_centers", "query_balancing", "num_points", "num_edges"])
normal_table = normal_table.sort_values(by=["num_procs", "method", "query_balancing"])
normal_table.round(3).to_csv("normal_results/table.csv")

rows = []

for path in Path("corel_results").glob("*.json"):
    fname = str(path)
    rows.append(read_stats_file(fname))

corel_table = pd.DataFrame(rows, columns=["infile", "method", "num_procs", "runtime", "dist_comps", "num_centers", "query_balancing", "num_points", "num_edges"])
corel_table = corel_table.sort_values(by=["num_procs", "method", "query_balancing"])
corel_table.round(3).to_csv("corel_results/table.csv")

rows = []

for path in Path("covtype_results").glob("*.json"):
    fname = str(path)
    rows.append(read_stats_file(fname))

covtype_table = pd.DataFrame(rows, columns=["infile", "method", "num_procs", "runtime", "dist_comps", "num_centers", "query_balancing", "num_points", "num_edges"])
covtype_table = covtype_table.sort_values(by=["num_procs", "method", "query_balancing"])
covtype_table.round(3).to_csv("covtype_results/table.csv")

for path in Path("blob_results").glob("gvor.steal*.json"):
    fname = str(path)
    num_procs, table = steal_stats_table(fname)
    table.round(3).to_csv(f"blob_results/steal.table.n{num_procs}.csv")

for path in Path("normal_results").glob("gvor.steal*.json"):
    fname = str(path)
    num_procs, table = steal_stats_table(fname)
    table.round(3).to_csv(f"normal_results/steal.table.n{num_procs}.csv")

for path in Path("covtype_results").glob("gvor.steal*.json"):
    fname = str(path)
    num_procs, table = steal_stats_table(fname)
    table.round(3).to_csv(f"covtype_results/steal.table.n{num_procs}.csv")

for path in Path("corel_results").glob("gvor.steal*.json"):
    fname = str(path)
    num_procs, table = steal_stats_table(fname)
    table.round(3).to_csv(f"corel_results/steal.table.n{num_procs}.csv")
