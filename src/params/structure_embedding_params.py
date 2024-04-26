import argparse


class StructureEmbeddingParams:

    def __init__(self):

        parser = argparse.ArgumentParser()
        parser.add_argument('--learning_rate', type=float)
        parser.add_argument('--weight_decay', type=float)
        parser.add_argument('--warmup_epochs', type=int)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument('--epochs', type=int)
        parser.add_argument('--input_layer', type=int)
        parser.add_argument('--dim_feedforward', type=int)
        parser.add_argument('--num_layers', type=int)
        parser.add_argument('--nhead', type=int)
        parser.add_argument('--hidden_layer', type=int)

        parser.add_argument('--test_every_n_steps', type=int)
        parser.add_argument('--devices', type=int)

        parser.add_argument('--class_path', required=True)
        parser.add_argument('--embedding_path', required=True)

        parser.add_argument('--metadata', type=str)

        args = parser.parse_args()

        self.learning_rate = args.learning_rate if args.learning_rate else 1e-6
        self.weight_decay = args.weight_decay if args.weight_decay else 0.
        self.warmup_epochs = args.warmup_epochs if args.warmup_epochs else 0
        self.batch_size = args.batch_size if args.batch_size else 32
        self.epochs = args.epochs if args.epochs else 100
        self.input_layer = args.input_layer if args.input_layer else 640
        self.dim_feedforward = args.dim_feedforward if args.dim_feedforward else self.input_layer
        self.nhead = args.nhead if args.nhead else 8
        self.num_layers = args.num_layers if args.num_layers else 6
        self.hidden_layer = args.hidden_layer if args.hidden_layer else self.input_layer

        self.test_every_n_steps = args.test_every_n_steps if args.test_every_n_steps else 10000
        self.devices = args.devices if args.devices else 1

        self.class_path = args.class_path
        self.embedding_path = args.embedding_path

        self.metadata = args.metadata if args.metadata else "None"

    def text_params(self, params=None):
        if params is None:
            params = {}
        return '\n'.join([
            "batch-size: %s  ",
            "learning-rate: %s  ",
            "weight-decay: %s  ",
            "warmup-epochs: %s  ",
            "input-layer: %s  ",
            "feed-forward: %s  ",
            "hidden-layer: %s  ",
            "num-layers: %s  ",
            "n-head: %s  ",
            "test-every-n-steps: %s  ",
            "metadata: %s  "
        ]) % (
            self.batch_size,
            self.learning_rate,
            self.weight_decay,
            self.warmup_epochs,
            self.input_layer,
            self.dim_feedforward,
            self.hidden_layer,
            self.num_layers,
            self.nhead,
            self.test_every_n_steps,
            self.metadata
        ) + "  \n" + "\n".join(["%s: %s" % (k, v) for k, v in params])
