
import logging
import os

import pandas as pd
from torch_geometric.data import Dataset

from dataset.utils.embedding_builder import graph_coords_builder

logger = logging.getLogger(__name__)


class GraphFromCoordsDataset(Dataset):
    def __init__(
            self,
            graph_list,
            output_path,
            postfix,
            coords_getter,
            num_workers=1,
            graph_builder=graph_coords_builder
    ):
        super().__init__()
        self.graphs = None
        self.nun_workers = num_workers if num_workers > 1 else 1
        self.graph_builder = graph_builder
        self.get_coords = coords_getter
        self.load_coords(graph_list, output_path, postfix)

    def load_coords(self, graph_list, output_path, postfix):
        logger.info(f"Loading graphs from file {graph_list}")
        self.graphs = pd.DataFrame(
            data=[
                (
                    graph_id
                ) for graph_id in [row.strip() for row in open(graph_list)]
                if not os.path.isfile(os.path.join(output_path, f"{graph_id}.{postfix}"))
            ],
            columns=['graph_id']
        )
        logger.info(f"Total graphs: {len(self.graphs)}")

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        graph = self.graphs.loc[idx]
        coords = self.get_coords(graph['graph_id'])
        out = self.graph_builder(
            [{'cas': ch[0], 'seq': ch[1]} for ch in zip(coords['cas'], coords['seq'])],
            num_workers=self.nun_workers
        )
        return out, graph['graph_id']
