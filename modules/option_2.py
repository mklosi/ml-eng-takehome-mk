from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import AdamW
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim


class RedditDataset(Dataset):
    def __init__(self, titles, bodies, scores, tokenizer, max_length):
        self.titles = titles
        self.bodies = bodies
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        # Combine title and body
        text = self.titles[idx] + " " + self.bodies[idx]
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "target": torch.tensor(self.scores[idx], dtype=torch.float),
        }


def main():

    # Example usage
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = RedditDataset(
        titles=["Post title 1", "Post title 2"],
        bodies=["Body 1", "Body 2"],
        scores=[3.5, 2.0],
        tokenizer=tokenizer,
        max_length=512,
    )

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=1  # For regression
    )

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(3):  # Number of epochs
        model.train()
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)  # Shape: (batch_size, 1)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=targets)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item()}")

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)

            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = outputs.logits  # Model outputs logits in regression
            print(f"Predictions: {predictions.squeeze()}, Targets: {targets.squeeze()}")


if __name__ == '__main__':
    main()
