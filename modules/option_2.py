from sklearn.neural_network import MLPRegressor

from modules.common import RedditScorePredictorSimpleBase
from modules.memory import Memory

mem = Memory(noop=False)


class RedditScorePredictorSimpleNN(RedditScorePredictorSimpleBase):

    model_path = "model_artifacts/model_option_2_nn.pkl"

    def __init__(self):
        super().__init__()

        self.model = MLPRegressor(
            hidden_layer_sizes=[100, 50],
            activation="relu",
            solver="adam",
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            verbose=False,
        )


if __name__ == '__main__':
    RedditScorePredictorSimpleNN.training_pipeline()
