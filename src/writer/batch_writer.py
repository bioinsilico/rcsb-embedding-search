from abc import abstractmethod
from collections import deque
from abc import ABC
import polars as pl
import torch

from lightning.pytorch.callbacks import BasePredictionWriter


class CoreBatchWriter(BasePredictionWriter, ABC):
    def __init__(
            self,
            output_path,
            write_interval="batch"
    ):
        super().__init__(write_interval)
        self.out_path = output_path

    def write_on_batch_end(
            self,
            trainer,
            pl_module,
            prediction,
            batch_indices,
            batch,
            batch_idx,
            dataloader_idx
    ):
        embeddings, dom_ids = prediction
        deque(map(
            self._write_embedding,
            embeddings,
            dom_ids
        ))

    def file_name(self, dom_id):
        return f'{self.out_path}/{dom_id}.pt'

    @abstractmethod
    def _write_embedding(self, embedding, dom_id):
        pass


class CsvBatchWriter(CoreBatchWriter, ABC):
    def __init__(
            self,
            output_path,
            write_interval="batch"
    ):
        super().__init__(output_path, write_interval)

    def _write_embedding(self, embedding, dom_id):
        pl.DataFrame(embedding.to('cpu').numpy()).write_csv(
            self.file_name(dom_id),
            include_header=False,
        )


class TensorBatchWriter(CoreBatchWriter, ABC):
    def __init__(
            self,
            output_path,
            write_interval="batch",
            device="cpu"
    ):
        super().__init__(output_path, write_interval)
        self.device = device

    def _write_embedding(self, embedding, dom_id):
        torch.save(
            embedding.to(self.device),
            f'{self.out_path}/{dom_id}.pt'
        )
