import logging
import lightning as L
from esm.models.esm3 import ESM3
from esm.sdk.api import SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL


from dataset.utils.tools import collate_seq_embeddings

logger = logging.getLogger(__name__)

esm_model = ESM3.from_pretrained(ESM3_OPEN_SMALL)


def esm_embeddings(esm_prot):
    protein_tensor = esm_model.encode(esm_prot)
    output = esm_model.forward_and_sample(
        protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
    )
    return output.per_residue_embedding


class LitStructureBiotiteEmbeddingPooling(L.LightningModule):

    def __init__(
            self,
            nn_model
    ):
        super().__init__()
        self.model = nn_model

    def predict_step(self, esm_prot_list, batch_idx):
        (x, x_mask) = collate_seq_embeddings([esm_embeddings(esm_prot) for esm_prot, acc in esm_prot_list])
        dom_id = tuple([acc for esm_prot, acc in esm_prot_list])
        return self.model.embedding_pooling(x, x_mask), dom_id
