from datetime import datetime

from modules.memory import Memory
from modules.option_1 import RedditScorePredictorSimpleRandomForest
from modules.option_2 import RedditScorePredictorSimpleNN
from modules.option_3 import RedditScorePredictorAdvanced

mem = Memory(noop=False)


def main():

    ## Choose your model class to predict with. Comment in the choice.

    ## This is ugly, I admit, but no time to change.
    # model_class = RedditScorePredictorSimpleRandomForest
    # model_class = RedditScorePredictorSimpleNN
    model_class = RedditScorePredictorAdvanced

    predictor = model_class.load_model(model_class.model_path)

    new_post = {
        "title": "Why is the sky blue?",
        "body": "This is a detailed explanation about Rayleigh scattering.",
        "tag": "Physics",
        "upvote_ratio": 0.9,
    }

    mem.log_memory(print, "prediction_start")
    prediction_dt = datetime.now()
    predictions = predictor.predict(new_post)
    mem.log_memory(print, "prediction_end")
    print(f"prediction runtime: {datetime.now() - prediction_dt}")

    print(f"Predicted Scores: {predictions}")


if __name__ == '__main__':
    main()
