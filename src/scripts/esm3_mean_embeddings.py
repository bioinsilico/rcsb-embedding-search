import argparse
import os

import pandas as pd
import torch

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    args = parser.parse_args()

    for e in os.listdir(args.embedding_path):
        embedding = torch.load(f"{args.embedding_path}/{e}").mean(dim=0)
        pd.DataFrame(embedding.to('cpu').numpy()).to_csv(
            f"{args.out_path}/{e.replace('pt','csv')}",
            index=False,
            header=False
        )

