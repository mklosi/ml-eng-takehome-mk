import joblib
import pandas as pd
import spacy  # Using spaCy for text cleaning
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class RedditScorePredictor:
    def __init__(self, model_artifact_path="reddit_model.pkl"):
        # Initialize all necessary components during instantiation
        self.model_artifact_path = model_artifact_path
        self.vectorizer = TfidfVectorizer(max_features=5000)  # TF-IDF vectorizer
        self.one_hot_encoder = OneHotEncoder(handle_unknown="ignore")  # For 'tag'
        self.scaler = StandardScaler()  # For 'upvote_ratio'
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)  # Random Forest regressor
        self.nlp = spacy.load("en_core_web_sm")  # SpaCy model for text cleaning

    # Class method to load a saved model and return a class instance
    @classmethod
    def load_model(cls, model_path):
        artifacts = joblib.load(model_path)
        instance = cls(model_artifact_path=model_path)
        instance.model = artifacts["model"]
        instance.vectorizer = artifacts["vectorizer"]
        instance.one_hot_encoder = artifacts["encoder"]
        instance.scaler = artifacts["scaler"]
        return instance

    # Class method to instantiate a new model
    @classmethod
    def new_model(cls, model_artifact_path="reddit_model.pkl"):
        return cls(model_artifact_path=model_artifact_path)

    # Method to save the current model to a file
    def save_model(self, save_path):
        joblib.dump(
            {
                "model": self.model,
                "vectorizer": self.vectorizer,
                "encoder": self.one_hot_encoder,
                "scaler": self.scaler,
            },
            save_path,
        )
        print(f"Model saved to {save_path}")

    # Method to load data from a CSV file
    def load_data(self, file_path):
        return pd.read_csv(file_path)

    # Method to clean text using spaCy
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        return " ".join(tokens)

    # Method to clean tabular features
    def clean_tabular_features(self, data):
        # Clean 'tag'
        data["tag"] = data["tag"].fillna("Unknown")
        # Ensure 'upvote_ratio' is numeric and fill missing values
        data["upvote_ratio"] = pd.to_numeric(data["upvote_ratio"], errors="coerce")
        data["upvote_ratio"] = data["upvote_ratio"].fillna(0.5)  # Default to 0.5
        return data

    # Method to combine text embeddings with tabular data
    def prepare_features(self, data, is_training=True):
        # Combine and clean text
        data["full_text"] = (data["title"].fillna("") + " " + data["body"].fillna("")).apply(self.clean_text)

        # Create text embeddings using TF-IDF
        if is_training:
            text_embeddings = self.vectorizer.fit_transform(data["full_text"])
        else:
            text_embeddings = self.vectorizer.transform(data["full_text"])

        # One-hot encode 'tag'
        if is_training:
            tag_embeddings = self.one_hot_encoder.fit_transform(data[["tag"]])
        else:
            tag_embeddings = self.one_hot_encoder.transform(data[["tag"]])

        # Standardize 'upvote_ratio'
        if is_training:
            upvote_ratios = self.scaler.fit_transform(data[["upvote_ratio"]])
        else:
            upvote_ratios = self.scaler.transform(data[["upvote_ratio"]])

        # Combine all features
        from scipy.sparse import hstack
        features = hstack([text_embeddings, tag_embeddings, upvote_ratios])
        return features

    # Method to train the model
    def train(self, data):
        data = self.clean_tabular_features(data)
        features = self.prepare_features(data, is_training=True)
        target = data["score"]

        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

        # Train the Random Forest model
        self.model.fit(X_train, y_train)

        # Validate the model
        predictions = self.model.predict(X_val)
        mse = mean_squared_error(y_val, predictions)
        print(f"Validation MSE: {mse}")

    # Method to validate the model
    def validate(self, data):
        data = self.clean_tabular_features(data)
        features = self.prepare_features(data, is_training=False)
        target = data["score"]

        # Predict and calculate MSE
        predictions = self.model.predict(features)
        mse = mean_squared_error(target, predictions)
        print(f"Validation MSE: {mse}")

    # Method for inference
    def predict(self, new_data):
        new_data = self.clean_tabular_features(new_data)
        features = self.prepare_features(new_data, is_training=False)
        predictions = self.model.predict(features)
        return predictions


def main():

    model_path = "trained_model_option_1.pkl"
    data_path = "askscience_data.csv"

    # Creating a New Model and Training
    predictor = RedditScorePredictor.new_model()
    data = predictor.load_data(data_path)
    predictor.train(data)
    predictor.save_model(model_path)

    # Loading a Saved Model
    loaded_predictor = RedditScorePredictor.load_model(model_path)
    data = loaded_predictor.load_data(data_path)
    loaded_predictor.validate(data)

    # Making Predictions
    loaded_predictor = RedditScorePredictor.load_model("trained_model.pkl")
    new_data = pd.DataFrame({
        "title": ["Why is the sky blue?"],
        "body": ["This is a detailed explanation about Rayleigh scattering."],
        "tag": ["Physics"],
        "upvote_ratio": [0.9]
    })
    predictions = loaded_predictor.predict(new_data)
    print(f"Predicted Score: {predictions[0]}")


if __name__ == '__main__':
    main()
