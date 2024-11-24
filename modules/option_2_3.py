import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW


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
        # Tokenize the input text
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


class BERTWithLSTM(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", hidden_dim=128, num_classes=1):
        super(BERTWithLSTM, self).__init__()
        # Load the pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        # Define an LSTM layer to process token-level embeddings
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        # Final dense layer for regression
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional LSTM

    def forward(self, input_ids, attention_mask):
        # Get token embeddings from BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # Shape: (batch_size, num_tokens, 768)

        # Pass token embeddings through LSTM
        lstm_out, (hidden, cell) = self.lstm(token_embeddings)
        # Aggregate the final hidden state from both directions
        final_hidden_state = torch.cat((hidden[0], hidden[1]), dim=1)  # Shape: (batch_size, hidden_dim * 2)

        # Pass the final hidden state through the dense layer for prediction
        predictions = self.fc(final_hidden_state)  # Shape: (batch_size, num_classes)
        return predictions


def main():
    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    dataset = RedditDataset(
        titles=["Post title 1", "Post title 2"],
        bodies=["Body 1", "Body 2"],
        scores=[3.5, 2.0],  # Example scores
        tokenizer=tokenizer,
        max_length=512,
    )

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Initialize the model
    model = BERTWithLSTM()

    # Define the optimizer
    # fdjkfjdkdfj
    # fjdkfjdk
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Define the loss function
    loss_fn = nn.MSELoss()

    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    for epoch in range(3):  # Number of epochs
        model.train()  # Set model to training mode
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)  # Shape: (batch_size, 1)

            # Forward pass
            output = model(input_ids, attention_mask=attention_mask)
            # missing logits

            # Compute the loss
            loss = loss_fn(output, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Evaluation loop
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)

            # Forward pass
            output = model(input_ids, attention_mask=attention_mask)
            print(f"Predictions: {output.squeeze().tolist()}, Targets: {targets.squeeze().tolist()}")


if __name__ == '__main__':
    main()
