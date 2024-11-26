import random
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

from modules.common import RedditScorePredictorBase, data_path
from modules.memory import Memory

mem = Memory(noop=False)


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seeds()


class RedditDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, is_training=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        inputs = self.tokenizer(
            row["full_text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }
        tabular_features = row[["tag_encoded", "upvote_ratio"]].astype(float).values
        item["tabular_features"] = torch.tensor(tabular_features, dtype=torch.float32)
        if self.is_training:
            item["score"] = torch.tensor(float(row["score"]), dtype=torch.float32)
        return item


class RedditScoreModule(nn.Module):
    def __init__(self, bert_model_name):
        super(RedditScoreModule, self).__init__()
        hidden_dim = 128
        tabular_input_dim = 2
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.fc1 = nn.Linear(self.bert.config.hidden_size + tabular_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, tabular_features):
        # Get the [CLS] embedding from BERT
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = bert_outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, 768)

        # Combine with tabular features
        combined_features = torch.cat((cls_embedding, tabular_features), dim=1)

        # Feed into dense layers
        x = self.fc1(combined_features)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x.squeeze(1)  # Output shape: (batch_size,)


class RedditScorePredictorAdvanced(RedditScorePredictorBase):

    model_path = "model_artifacts/model_option_3.pkl"

    def __init__(self):
        self.bert_model_name = "bert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
        self.model = RedditScoreModule(bert_model_name=self.bert_model_name)
        self.one_hot_encoder = OneHotEncoder(handle_unknown="ignore")
        self.scaler = StandardScaler()
        self.max_length = 512
        self.batch_size = 16
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)  # default: 0.0001
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.to(self.device)

    @classmethod
    def load_model(cls, model_path):
        artifacts = joblib.load(model_path)
        instance = cls()
        instance.model.load_state_dict(artifacts["model"])
        instance.tokenizer = BertTokenizer.from_pretrained(artifacts["tokenizer"])
        instance.one_hot_encoder = artifacts["encoder"]
        instance.scaler = artifacts["scaler"]
        print(f"Model loaded from: {model_path}")
        return instance

    def save_model(self, save_path):
        artifacts = {
            "model": self.model.state_dict(),
            "tokenizer": self.bert_model_name,
            "encoder": self.one_hot_encoder,
            "scaler": self.scaler,
        }
        joblib.dump(artifacts, save_path)
        print(f"Model saved to: {save_path}")

    def prepare_features(self, data, is_training=True):
        print("Preparing features...")

        # Only use 'tag' and 'upvote_ratio' as tabular features for now. Remove the others.
        data = data[["title", "body", "tag", "upvote_ratio", "score"]].copy()
        data["tag"] = data["tag"].fillna("Unknown")
        data["upvote_ratio"] = pd.to_numeric(data["upvote_ratio"], errors="coerce").fillna(0.5)
        data["full_text"] = (data["title"].fillna("") + " " + data["body"].fillna("")).apply(self.clean_text)

        if is_training:
            tag_encoded = self.one_hot_encoder.fit_transform(data[["tag"]]).toarray()[:, 0]
            data["tag_encoded"] = pd.to_numeric(tag_encoded, errors="coerce").astype(float)
            data["upvote_ratio"] = pd.to_numeric(self.scaler.fit_transform(data[["upvote_ratio"]]).flatten(), errors="coerce")
        else:
            tag_encoded = self.one_hot_encoder.transform(data[["tag"]]).toarray()[:, 0]
            data["tag_encoded"] = pd.to_numeric(tag_encoded, errors="coerce").astype(float)
            data["upvote_ratio"] = pd.to_numeric(self.scaler.transform(data[["upvote_ratio"]]).flatten(), errors="coerce")

        return data

    def train(self, train_data):

        print("Training model...")
        train_dt = datetime.now()

        epochs = 5  # move this as instance var.

        train_data, val_data = self.split_data(train_data)

        train_data = self.prepare_features(train_data, is_training=True)
        val_data = self.prepare_features(val_data, is_training=False)

        train_dataset = RedditDataset(train_data, self.tokenizer, self.max_length, is_training=True)
        val_dataset = RedditDataset(val_data, self.tokenizer, self.max_length, is_training=True)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        for epoch in range(epochs):
            print(f"--- Epoch '{epoch + 1}/{epochs}' --------------------")
            self.model.train()
            epoch_loss = 0
            for idx, batch in enumerate(train_loader):
                print(f"Processing batch: {idx+1}/{len(train_loader)}")
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                tabular_features = batch["tabular_features"].to(self.device)
                scores = batch["score"].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask, tabular_features)
                loss = self.criterion(outputs, scores)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            print(f"Training Loss:   {epoch_loss / len(train_loader)}")

            self.validate(val_loader)

        mem.log_memory(print, "train")
        print(f"train runtime: {datetime.now() - train_dt}")

    def validate(self, data_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                tabular_features = batch["tabular_features"].to(self.device)
                scores = batch["score"].to(self.device)

                outputs = self.model(input_ids, attention_mask, tabular_features)
                loss = self.criterion(outputs, scores)
                total_loss += loss.item()

        print(f"Validation Loss: {total_loss / len(data_loader)}")

    def test(self, test_data):
        test_data = self.prepare_features(test_data, is_training=False)
        test_dataset = RedditDataset(test_data, self.tokenizer, self.max_length, is_training=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                tabular_features = batch["tabular_features"].to(self.device)
                scores = batch["score"].to(self.device)

                outputs = self.model(input_ids, attention_mask, tabular_features)
                loss = self.criterion(outputs, scores)
                total_loss += loss.item()

        mean_loss = total_loss / len(test_loader)
        print(f"Test Loss (MSE): {mean_loss}")
        return {
            "mse": mean_loss,
        }

    def predict_batch(self, new_data):

        # This is not used during prediction. It's here just as a fast
        #   and easy way to get 'prepare_features' func to stop complaining.
        new_data["score"] = None

        new_data = self.prepare_features(new_data, is_training=False)
        dataset = RedditDataset(new_data, self.tokenizer, self.max_length, is_training=False)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                tabular_features = batch["tabular_features"].to(self.device)

                outputs = self.model(input_ids, attention_mask, tabular_features)
                predictions.extend(outputs.cpu().numpy())

        return predictions

    @classmethod
    def training_pipeline(cls):
        # Create a New Model.
        predictor = cls.new_model()

        # Load the data.
        data = predictor.load_data(data_path)

        # Take only subset of data. For testing only. This is to be commented out.
        data = data.head(100)  # for option 3.

        # Regular train on a single fold, test, and save a model.
        train_data, test_data = predictor.split_data(data)
        predictor.train(train_data)
        predictor.test(test_data)
        predictor.save_model(cls.model_path)

        # Loading a Saved Model and test again.
        predictor = cls.load_model(cls.model_path)
        predictor.test(test_data)


if __name__ == "__main__":
    RedditScorePredictorAdvanced.training_pipeline()
