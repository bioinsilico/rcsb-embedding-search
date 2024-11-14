import gzip
import logging
import pickle

from torch_geometric.data import Dataset

from dataset.utils.embedding_builder import graph_ch_coords_builder

logger = logging.getLogger(__name__)


class GraphFromPklDataset(Dataset):
    def __init__(
            self,
            pkl_file,
            num_workers=1,
            graph_builder=graph_ch_coords_builder
    ):
        super().__init__()
        self.graphs = None
        self.nun_workers = num_workers if num_workers > 1 else 1
        self.graph_builder = graph_builder
        self.load_coords(pkl_file)

    def load_coords(self, pkl_file):
        logger.info(f"Loading coords from file {pkl_file}")
        with gzip.open(pkl_file, 'rb') as f:
            self.graphs = pickle.load(f)
        logger.info(f"Total graphs: {len(self.graphs)}")

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        graph = self.graphs[idx]
        out = self.graph_builder(
            [{'cas': cas, 'seq': seq} for (cas, seq) in zip(graph['cas'], graph['seq'])],
            num_workers=self.nun_workers
        )
        return out, graph['id']
