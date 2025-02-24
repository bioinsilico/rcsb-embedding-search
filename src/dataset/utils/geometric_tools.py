import math
import torch

from dataset.utils.geometry import angle_between_points, exp_distance, distance_between_points, angle_between_planes, \
    angle_between_four_points


def get_angles(idx, residues):
    if 0 < idx < len(residues) - 1:
        res_0, idx_0 = residues[idx]
        res_f, idx_f = residues[idx + 1]
        res_b, idx_b = residues[idx - 1]
        if idx_0 - idx_b == 1 and idx_f - idx_0 == 1:
            a = angle_between_points(res_b, res_0, res_f)
            return (
                math.sin(a),
                math.cos(a)
            )
    return 0, 0


def get_distances(idx, residues):
    df = 0
    db = 0
    if idx < len(residues) - 1:
        res_0, idx_0 = residues[idx]
        res_f, idx_f = residues[idx + 1]
        if idx_f - idx_0 == 1:
            df = exp_distance(distance_between_points(res_f, res_0))
    if idx > 0:
        res_0, idx_0 = residues[idx]
        res_b, idx_b = residues[idx -1]
        if idx_0 - idx_b == 1:
            db = exp_distance(distance_between_points(res_0, res_b))
    return df, db


def contiguous(idx_i, idx_j):
    if abs(idx_i - idx_j) == 1:
        return 1
    return 0


def edge_angles(idx_i, idx_j, residues):
    if (
            (0 < idx_i < len(residues) - 1 and 0 < idx_j < len(residues) - 1) and
            (residues[idx_i][1] - residues[idx_i-1][1] == 1) and
            (residues[idx_i+1][1] - residues[idx_i][1] == 1) and
            (residues[idx_j][1] - residues[idx_j-1][1] == 1) and
            (residues[idx_j+1][1] - residues[idx_j][1] == 1)
    ):
        a = angle_between_planes(
            (residues[idx_i-1][0], residues[idx_i][0], residues[idx_i+1][0]),
            (residues[idx_j-1][0], residues[idx_j][0], residues[idx_j+1][0])
        )
        return (
            math.sin(a),
            math.cos(a)
        )
    return 0, 0


def orientation_angles(idx_i, idx_j, residues):
    fs = 0
    fc = 0
    bs = 0
    bc = 0
    if (
            (idx_i < len(residues) - 1) and
            (idx_j < len(residues) - 1) and
            (residues[idx_i + 1][1] - residues[idx_i][1] == 1) and
            (residues[idx_j + 1][1] - residues[idx_j][1] == 1)
    ):
        a = angle_between_four_points(
            residues[idx_i][0], residues[idx_i + 1][0],
            residues[idx_j][0], residues[idx_j + 1][0]
        )
        fs = math.sin(a)
        fc = math.cos(a)
    if (
            (idx_i > 0 and idx_j > 0) and
            (residues[idx_i][1] - residues[idx_i-1][1] == 1) and
            (residues[idx_j][1] - residues[idx_j-1][1] == 1)
    ):
        a = angle_between_four_points(
            residues[idx_i - 1][0], residues[idx_i][0],
            residues[idx_j - 1][0], residues[idx_j][0]
        )
        bs = math.sin(a)
        bc = math.cos(a)
    return fs, fc, bs, bc


def get_contacts(residues):
    contact_map = []
    for i, (res_i, idx_i) in enumerate(residues):
        for j, (res_j, idx_j) in enumerate(residues):
            if i == j:
                continue
            d = distance_between_points(res_i, res_j)
            if d < 8.:
                e = exp_distance(d)
                c = contiguous(idx_i, idx_j)
                es, ec = edge_angles(i, j, residues)
                fs, fc, bs, bc = orientation_angles(i, j, residues)
                contact_map.append(([i, j], (e, c, es, ec, fs, fc, bs, bc)))
    return contact_map


def get_res_attr(residues):
    angles = [get_angles(idx, residues) for idx, res in enumerate(residues)]
    distances = [get_distances(idx, residues) for idx, res in enumerate(residues)]
    contacts = get_contacts(residues)
    graph_nodes = torch.tensor([[
        a, b, df, db
    ] for (a, b), (df, db) in list(zip(angles, distances))], dtype=torch.float)
    graph_edges = torch.tensor([
        c for c, d in contacts
    ], dtype=torch.int64)
    edge_attr = torch.tensor([
        d for c, d in contacts
    ])
    return graph_nodes, graph_edges, edge_attr

