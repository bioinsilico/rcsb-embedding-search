import argparse
import os

import pandas as pd
import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader


class GraphFromListDataset(Dataset):
    def __init__(
            self,
            graph_list,
            graph_path,
            output_path,
            postfix
    ):
        super().__init__()
        self.graphs = None
        self.load_coords(graph_list, graph_path, output_path, postfix)

    def load_coords(self, graph_list, graph_path, output_path, postfix):
        self.graphs = pd.DataFrame(
            data=[
                (
                    graph_id,
                    os.path.join(graph_path, f"{graph_id}.pt")
                ) for graph_id in [row.strip() for row in open(graph_list)]
                if not os.path.isfile(os.path.join(output_path, f"{graph_id}.{postfix}"))
            ],
            columns=['graph_id', 'graph_file']
        )
        print(f"Total graphs: {len(self.graphs)}")

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        graph = self.graphs.loc[idx]
        return torch.load(graph['graph_file']), graph['graph_id']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_list', type=str, required=True)
    parser.add_argument('--graph_path', type=str, required=True)
    args = parser.parse_args()

    dataset = GraphFromListDataset(
        args.graph_list,
        args.graph_path,
        output_path="/tmp",
        postfix="none"
    )
    dataloader = DataLoader(
        dataset,
        batch_size=8
    )
    for g, g_id in dataloader:
        print(g_id)

