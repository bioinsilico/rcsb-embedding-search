import argparse

import torch

from lightning_module.inference.embedding_inference import LitEmbeddingInference
from networks.transformer_nn import TransformerEmbeddingCosine

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    args = parser.parse_args()

    nn_model = TransformerEmbeddingCosine(
        input_features=1536,
        nhead=12,
        num_layers=6,
        dim_feedforward=3072,
        hidden_layer=1536,
        res_block_layers=12
    )

    model = LitEmbeddingInference.load_from_checkpoint(
        args.checkpoint,
        nn_model=nn_model
    )

    torch.save(model.state_dict(), args.out_file)
