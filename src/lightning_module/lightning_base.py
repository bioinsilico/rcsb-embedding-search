import lightning as L
from torch import nn, optim, cat
from torcheval.metrics.functional import binary_auprc, binary_auroc

from src.lightning_module.utils import get_cosine_schedule_with_warmup


class LitStructureBase(L.LightningModule):
    PR_AUC_METRIC_NAME = 'pr_auc'

    def __init__(
            self,
            nn_model,
            learning_rate=1e-6,
            params=None
    ):
        super().__init__()
        self.model = nn_model
        self.learning_rate = learning_rate
        self.params = params
        self.z = []
        self.z_pred = []

    def on_fit_start(self):
        if self.params and hasattr(self.logger.experiment, 'add_text'):
            summary = L.pytorch.utilities.model_summary.LayerSummary(self)
            self.logger.experiment.add_text(
                "Param Description",
                self.params.text_params({
                    "Number-parameters": f"{summary.num_parameters // 1e6}M"
                })
            )

    def on_train_epoch_start(self):
        self.reset_z()

    def on_train_epoch_end(self):
        z = cat(self.z, dim=0)
        z_pred = cat(self.z_pred, dim=0)
        loss = nn.functional.mse_loss(z_pred, z)
        self.log("train_loss", loss, sync_dist=True)

    def on_validation_epoch_start(self):
        self.reset_z()

    def on_validation_epoch_end(self):
        z = cat(self.z, dim=0)
        z_pred = cat(self.z_pred, dim=0)
        pr_auc = binary_auprc(z_pred, z)
        self.log(self.PR_AUC_METRIC_NAME, pr_auc, sync_dist=True)
        if self.device.type == 'mps':
            roc_auc = binary_auroc(z_pred.to('cpu'), z.to('cpu'))
        else:
            roc_auc = binary_auroc(z_pred, z)
        self.log("roc_auc", roc_auc, sync_dist=True)

    def reset_z(self):
        self.z = []
        self.z_pred = []
