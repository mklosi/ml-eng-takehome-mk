from sklearn.ensemble import RandomForestRegressor

from modules.common import RedditScorePredictorSimpleBase
from modules.memory import Memory

mem = Memory(noop=False)


class RedditScorePredictorSimpleRandomForest(RedditScorePredictorSimpleBase):

    model_path = "model_artifacts/model_option_1_forest.pkl"

    def __init__(self):
        super().__init__()

        self.model = RandomForestRegressor(n_estimators=100, random_state=42, verbose=False)


if __name__ == '__main__':
    RedditScorePredictorSimpleRandomForest.training_pipeline()
