from datetime import datetime

import joblib
import pandas as pd
import torch
from cleantext import clean
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from modules.memory import Memory

print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.backends.mps.is_available(): {torch.backends.mps.is_available()}")

mem = Memory(noop=False)

data_path = "askscience_data.csv"


class RedditScorePredictorBase:

    # Class method to load a saved model and return a class instance
    @classmethod
    def load_model(cls, model_path):
        raise NotImplementedError()

    # Class method to instantiate a new model
    @classmethod
    def new_model(cls):
        return cls()

    # Method to save the current model to a file
    def save_model(self, save_path):
        raise NotImplementedError()

    @staticmethod
    def load_data(file_path):
        df = pd.read_csv(file_path)
        # # keeping the original index, since it seems to be out of order and a lot of duplicates.
        # df = df.rename(columns={"Unnamed: 0": "original_index"})
        df = df.drop(columns=["Unnamed: 0"])
        return df

    @staticmethod
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        cleaned_text = clean(
            text,
            clean_all=False,
            extra_spaces=True,
            stemming=True,
            stopwords=True,
            lowercase=True,
            numbers=False,
            punct=True,
        )
        return cleaned_text

    def prepare_features(self, data, is_training=True):
        raise NotImplementedError()

    @staticmethod
    def split_data(data):
        return train_test_split(data, test_size=0.2, random_state=42)

    def train(self, train_data):
        raise NotImplementedError()

    def test(self, test_data):
        raise NotImplementedError()

    # Method for inference
    def predict_batch(self, new_data):
        raise NotImplementedError()

    def predict(self, new_post):
        new_data = pd.DataFrame([new_post])
        return self.predict_batch(new_data)[0]


class RedditScorePredictorSimpleBase(RedditScorePredictorBase):
    def __init__(self):
        self.tokenizer = TfidfVectorizer(max_features=5000)
        self.one_hot_encoder = OneHotEncoder(handle_unknown="ignore")  # For 'tag'
        self.scaler = StandardScaler()  # For 'upvote_ratio'

    @classmethod
    def load_model(cls, model_path):
        artifacts = joblib.load(model_path)
        instance = cls()
        instance.model = artifacts["model"]
        instance.tokenizer = artifacts["tokenizer"]
        instance.one_hot_encoder = artifacts["encoder"]
        instance.scaler = artifacts["scaler"]
        print(f"Model loaded from: {model_path}")
        return instance

    def save_model(self, save_path):
        artifacts = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "encoder": self.one_hot_encoder,
            "scaler": self.scaler,
        }
        joblib.dump(artifacts, save_path)
        print(f"Model saved to: {save_path}")

    def prepare_features(self, data, is_training=True):
        print("Preparing features...")

        # Only use 'tag' and 'upvote_ratio' as tabular features for now. Remove the others.
        data = data[["title", "body", "tag", "upvote_ratio"]].copy()
        data["tag"] = data["tag"].fillna("Unknown")
        data["upvote_ratio"] = pd.to_numeric(data["upvote_ratio"], errors="coerce").fillna(0.5)
        data["full_text"] = (data["title"].fillna("") + " " + data["body"].fillna("")).apply(self.clean_text)

        # Create text embeddings using TF-IDF
        if is_training:
            tag_embeddings = self.one_hot_encoder.fit_transform(data[["tag"]])
            text_embeddings = self.tokenizer.fit_transform(data["full_text"])
            upvote_ratios = self.scaler.fit_transform(data[["upvote_ratio"]])
        else:
            tag_embeddings = self.one_hot_encoder.transform(data[["tag"]])
            text_embeddings = self.tokenizer.transform(data["full_text"])
            upvote_ratios = self.scaler.transform(data[["upvote_ratio"]])

        # Combine all features
        features = hstack([text_embeddings, tag_embeddings, upvote_ratios])

        return features

    def train(self, train_data):

        X_train = self.prepare_features(train_data, is_training=True)
        y_train = train_data["score"]

        print("Training model...")
        train_dt = datetime.now()
        self.model.fit(X_train, y_train)

        ## Comment this out, so we can more easily compare outputs between architectures. Enable as needed.
        # if hasattr(self.model, "loss_"):
        #     print(f"Training loss: {self.model.loss_}")

        mem.log_memory(print, "train")
        print(f"train runtime: {datetime.now() - train_dt}")

    def test(self, test_data):
        X_test = self.prepare_features(test_data, is_training=False)
        y_test = test_data["score"]

        # Predict and calculate MSE
        predictions = self.model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        rmse = root_mean_squared_error(y_test, predictions)
        print(f"Test MSE: {mse}")
        print(f"Test RMSE: {rmse}")
        return {
            "mse": mse,
            "rmse": rmse,
        }

    def train_with_cross_validation(self, data, folds):
        # TODO: docs that we can either use this or just train, but not both.
        # TODO: mention avoid using cross_val_score

        min_mse = 40_000_000
        kf = KFold(n_splits=folds, shuffle=True, random_state=42)
        fold_res = []

        print("Training model with cross-validation...")
        train_dt = datetime.now()
        for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
            print(f"--- Processing fold {fold + 1}/{folds}... -----------------------")
            train_data_fold = data.iloc[train_idx]
            val_data_fold = data.iloc[val_idx]

            self.train(train_data_fold)
            test_res = self.test(val_data_fold)

            fold_res.append(test_res)

        print("--- Summary performance metrics ------------------------")

        # Calculate the average MSE across folds
        fold_metric = [res["mse"] for res in fold_res]
        mean_metric = sum(fold_metric) / len(fold_metric)
        print(f"Cross-Validation MSE (mean over {folds} folds): {mean_metric}")

        # Very simple way to stop training, if min bar is not cleared.
        if mean_metric > min_mse:
            raise ValueError(f"'{self.__class__}' did not pass cross validation.")

        # Calculate the average MSE across folds
        fold_metric = [res["rmse"] for res in fold_res]
        mean_metric = sum(fold_metric) / len(fold_metric)
        print(f"Cross-Validation RMSE (mean over {folds} folds): {mean_metric}")

        print("--- Final training ------------------------")
        self.train(data)
        print(f"Total Cross-Validation training runtime: {datetime.now() - train_dt}")

    def predict_batch(self, new_data):
        features = self.prepare_features(new_data, is_training=False)
        predictions = self.model.predict(features)
        return predictions

    @classmethod
    def training_pipeline(cls):
        # Create a New Model.
        predictor = cls.new_model()

        # Load the data.
        data = predictor.load_data(data_path)

        # Take only subset of data. For testing only. This is to be commented out.
        data = data.head(1000)  # for option_1 and 2.

        # # Basic train on all data with no tests. This is only to retrain a model
        # #   whose architecture we've already previously discovered to be optimal. For testing only.
        # predictor.train(data)

        # # The following code is commented out, since we are doing cross-validation training later.
        # # Regular train on a single fold, test, and save a model.
        # train_data, test_data = predictor.split_data(data)
        # predictor.train(train_data)
        # predictor.test(test_data)
        # predictor.save_model(cls.model_path)
        #
        # # Loading a Saved Model and test again.
        # predictor = cls.load_model(cls.model_path)
        # predictor.test(test_data)

        # Train with cross-validation. Doesn't use `split_data`. Uses the full dataset.
        predictor.train_with_cross_validation(data, folds=5)
        predictor.save_model(predictor.model_path)
