import argparse


class StructureEmbeddingParams:

    def __init__(self):

        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--train_class_file', required=True)
        self.parser.add_argument('--train_embedding_path', required=True)
        self.parser.add_argument('--test_class_file', required=True)
        self.parser.add_argument('--test_embedding_path', required=True)

        self.parser.add_argument('--learning_rate', type=float)
        self.parser.add_argument('--weight_decay', type=float)
        self.parser.add_argument('--warmup_epochs', type=int)
        self.parser.add_argument('--batch_size', type=int)
        self.parser.add_argument('--epochs', type=int)
        self.parser.add_argument('--epoch_size', type=int)
        self.parser.add_argument('--input_layer', type=int)
        self.parser.add_argument('--dim_feedforward', type=int)
        self.parser.add_argument('--num_layers', type=int)
        self.parser.add_argument('--nhead', type=int)
        self.parser.add_argument('--amplify_input', type=int)
        self.parser.add_argument('--hidden_layer', type=int)

        self.parser.add_argument('--test_every_n_steps', type=int)
        self.parser.add_argument('--devices', type=int)
        self.parser.add_argument('--workers', type=int)
        self.parser.add_argument('--strategy', type=str)

        self.parser.add_argument('--checkpoint', type=str)
        self.parser.add_argument('--metadata', type=str)

        args = self.parser.parse_args()

        self.learning_rate = args.learning_rate if args.learning_rate else 1e-6
        self.weight_decay = args.weight_decay if args.weight_decay else 0.
        self.warmup_epochs = args.warmup_epochs if args.warmup_epochs else 0
        self.batch_size = args.batch_size if args.batch_size else 32
        self.epochs = args.epochs if args.epochs else 100
        self.epoch_size = args.epoch_size if args.epoch_size else 0
        self.input_layer = args.input_layer if args.input_layer else 640
        self.dim_feedforward = args.dim_feedforward if args.dim_feedforward else self.input_layer
        self.nhead = args.nhead if args.nhead else 8
        self.num_layers = args.num_layers if args.num_layers else 6
        self.hidden_layer = args.hidden_layer if args.hidden_layer else self.input_layer

        self.test_every_n_steps = args.test_every_n_steps if args.test_every_n_steps else 10000
        self.devices = args.devices if args.devices else 1
        self.workers = args.workers if args.workers else 2
        self.strategy = args.strategy if args.strategy else "auto"

        self.metadata = args.metadata if args.metadata else "None"
        self.checkpoint = args.checkpoint if args.checkpoint else "None"

        self.train_class_file = args.train_class_file
        self.train_embedding_path = args.train_embedding_path
        self.test_class_file = args.test_class_file
        self.test_embedding_path = args.test_embedding_path

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
        ) + "  \n" + "\n".join(["%s: %s" % (k, v) for k, v in params.items()])