class StructureEmbeddingParams:

    def __init__(self, args):
        self.learning_rate = args.learning_rate if args.learning_rate else 1e-6
        self.batch_size = args.batch_size if args.batch_size else 32
        self.epochs = args.epochs if args.epochs else 10
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
