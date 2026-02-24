from torch import nn, cat
import torch
from lightning_module.lightning_core import LitStructureCore


class LitEmbeddingTraining(LitStructureCore):

    def __init__(
            self,
            nn_model,
            learning_rate=1e-6,
            params=None,
            cov_reg_weight=1e-12,
            norm_reg_weight=1e-7
    ):
        super().__init__(nn_model, learning_rate, params)
        self.cov_reg_weight = cov_reg_weight
        self.norm_reg_weight = norm_reg_weight

    def covariance_loss(self, embeddings_x, embeddings_y):
        """
        Compute covariance regularization loss to decorrelate embedding dimensions.

        Args:
            embeddings_x: embeddings for first structure [batch_size, embedding_dim]
            embeddings_y: embeddings for second structure [batch_size, embedding_dim]

        Returns:
            Covariance loss (sum of squared off-diagonal covariance elements)
        """
        # Combine both embeddings for more samples
        embeddings = cat([embeddings_x, embeddings_y], dim=0)

        # Center the embeddings
        embeddings_centered = embeddings - embeddings.mean(dim=0, keepdim=True)

        # Compute covariance matrix
        batch_size = embeddings_centered.size(0)
        cov_matrix = (embeddings_centered.T @ embeddings_centered) / (batch_size - 1)

        # Penalize off-diagonal elements (encourage decorrelation)
        cov_loss = (cov_matrix ** 2).sum() - (torch.diag(cov_matrix) ** 2).sum()

        return cov_loss

    def norm_loss(self, embeddings_x, embeddings_y):
        """
        Penalize embeddings whose L2 norm deviates from 1.

        Args:
            embeddings_x: embeddings for first structure [batch_size, embedding_dim]
            embeddings_y: embeddings for second structure [batch_size, embedding_dim]

        Returns:
            Norm loss (mean squared deviation from unit norm)
        """
        norms_x = torch.norm(embeddings_x, p=2, dim=1)
        norms_y = torch.norm(embeddings_y, p=2, dim=1)
        norm_loss = ((norms_x - 1) ** 2).mean() + ((norms_y - 1) ** 2).mean()
        return norm_loss

    def training_step(self, batch, batch_idx):
        (x, x_mask), (y, y_mask), z = batch
        z_pred = self.model(x, x_mask, y, y_mask)
        self.z = cat((self.z, z), dim=0)
        self.z_pred = cat((self.z_pred, z_pred), dim=0)

        # MSE loss
        mse_loss = nn.functional.mse_loss(z_pred, z)

        # Covariance regularization
        embeddings_x = self.model.embedding_pooling(x, x_mask)
        embeddings_y = self.model.embedding_pooling(y, y_mask)
        cov_loss = self.covariance_loss(embeddings_x, embeddings_y)

        # Norm regularization
        norm_loss_val = self.norm_loss(embeddings_x, embeddings_y)

        # Total loss
        total_loss = mse_loss + self.cov_reg_weight * cov_loss + self.norm_reg_weight * norm_loss_val

        # Log losses
        self.log('train_mse_loss', mse_loss, prog_bar=True)
        self.log('train_cov_loss', cov_loss, prog_bar=True)
        self.log('train_norm_loss', norm_loss_val, prog_bar=True)
        self.log('train_total_loss', total_loss, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        (x, x_mask), (y, y_mask), z = batch
        z_pred = self.model(x, x_mask, y, y_mask)
        self.z = cat((self.z, z), dim=0)
        self.z_pred = cat((self.z_pred, z_pred), dim=0)
