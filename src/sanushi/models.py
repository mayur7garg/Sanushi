import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

class Sanushi(nn.Module):
    def __init__(self, node_count, input_dim, embedding_dim = 64):
        super(Sanushi, self).__init__()

        self.node_embedding = nn.Embedding(node_count, embedding_dim)
        self.non_linear = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.Sigmoid(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, input_array, node_idx):
        node_emb = self.node_embedding(node_idx)
        input_emb = self.non_linear(input_array)

        cosine_sim = F.cosine_similarity(input_emb, node_emb, dim = -1)
        return cosine_sim

    def get_similarities(self, input_array):
        with torch.no_grad():
            input_emb = self.non_linear(input_array)
            similarities = F.cosine_similarity(
                input_emb.unsqueeze(-2),
                self.node_embedding.weight.unsqueeze(0),
                dim = -1
            )
            return similarities.squeeze(0) if input_array.ndim == 1 else similarities

class SanushiTrainer:
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.001)
        self.criterion = nn.BCELoss()
        self.epochs_trained = 0

    def train(self, num_epochs = 10, log_freq = 1):
        self.model.train()

        for epoch in tqdm(range(num_epochs), desc = "Training"):
            total_loss = 0.0

            for batch_idx, (input_array, node_indices, targets) in tqdm(enumerate(self.data_loader), leave = False, total = len(self.data_loader), desc = "Batches"):
                self.optimizer.zero_grad()

                predictions = self.model(input_array, node_indices)
                predictions_clipped = torch.clamp(predictions, min = 1e-7, max = 1 - 1e-7)

                loss = self.criterion(predictions_clipped, targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.data_loader)
            self.epochs_trained += 1

            if epoch % log_freq == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")