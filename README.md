# Take-Home

We are looking for a candidate who can demonstrate their ability to learn new technologies and solve problems.

Please take this as an opportunity to showcase your skill. There are not necessarily any specific correct answers to the questions. We are looking to follow your thought process and how you work through a somewhat practical problem.

You are free to present your solution in any format you desire. Jupyter notebooks, Markdown docs, and/or Python modules are preferred, but not required.

Expected time spent: **4 hours**

## Submission Instructions

All submitted responses should be in the form of a pull request (PR), drawn from a __fork__ of this repository, as if you were contributing to an open-source project.  Be sure to follow good PR practices.

You are tasked with building a mini end-to-end ML pipeline to analyze and model data from Reddit posts on r/askscience. 

The file `askscience_data.csv` contains information about posts, including titles, bodies, scores, and metadata.  Your pipeline should generate a predicted Reddit post score, given at least a post title and body (and any other feature as you choose).

1. **Data Preprocessing Pipeline:**
   - Write reusable Python functions to:
     - Load and clean the dataset. This could involve basic string cleaning, standardizations like lowercasing, etc.
     - Transform the cleaned dataset into a format ready for modeling. For example, this could involve tokenization, embedding generation, etc.
   - Briefly explain your preprocessing choices
   
2. **Simple Predictive Model:**
   - Train and evaluate a model to predict the score of a post based on its features.  
   - You may use traditional classification models (e.g. regression/tree-based models), LLMs, or a combination thereof.
   - Wrap the model logic (training/evaluation/etc) into functions or classes that could fit into a production / training pipeline

3. **Simulated Deployment:**
   - Create a Python function to simulate the prediction or inference step for new data points
       - Example input: A dictionary containing the title and body of a post
       - Example output: A predicted score
   - Record any relevant metrics (e.g., inference time) to simulate MLOps tracking.
   
4. **Discussion**
   - Discuss and reflect on the performance of your model. What metrics did you use to evaluate it? How would you improve it?
   - Discuss MLOps metrics that you would track in a production deployment of this model:
        - For example, how would you monitor for model drift, inference latency issues, or data quality problems?
        - If you have experience with MLOps tools or frameworks, discuss how you would apply them in this scenario
   - If you did not use an LLM:
        - Explain why an LLM might or might not be appropriate for this task.
        - Discuss how you would integrate an LLM to improve feature extraction, predictions, or the pipeline overall.
   - If you did use an LLM:
        - Describe its impact on model performance.
        - Suggest specific ways to improve the LLM’s integration or enhance its performance with more time or resources.
     
If you have any questions please reach out to Yiming at yiming.liu@pivotallifesciences.com

## Submission

### Requirements 

1. Python 3.10

2. [poetry](https://python-poetry.org/) - Python env management tool. To install on a mac, run `brew install poerty`

3. [direnv](https://direnv.net/) - An extension for the shell. It augments existing shells with a new feature that can load and unload environment variables depending on the current directory. To install on a mac, run `brew install direnv`

### Set Up

Clone this branch and cd into repo's root dir. 'direnv' + 'poetry' should do the rest.

### Implementation Options

I've implemented this using three different methods. 

1. The first method is the simplest (option 1). It uses TF-IDF to vectorize the text (no LLMs) and scikit-learn's simple RandomForestRegressor model for training and inference. It combines the title and text of posts into a single text before vectorization. It uses two additional features ('tag' and 'upvote_ratio'). I combine the text tokens with the tabular features using scipy's hstack. Before feature combo, there's cleaning and transformations done to both text and tabular data. This model is the simplest, trains the fastest on my local laptop, and yet produces the best results. This is probably due to the fact that the training data is not that big (~4000 posts).

2. The next model is slightly more complicated (option 2). It uses almost all the code that option 1 uses, but it replaces the  RandomForestRegressor model with scikit-learn's MLPRegressor model.

For both option 1 and 2, I've used only scikit-learn tools.

3. The final model is the most advanced (option 3). It swaps scikit-learn with PyTorch. Then it uses Google's BERT tokenizer and model in order to create text embeddings. These embeddings are combined with the tabular feature data, similar to the previous options. The code differs substantially, although I try to keep the general method structure the same. Once feature tensors are ready, they are passed to a custom deep learning architecture that I have defined myself (RedditScoreModule). It's a simple densely-layered nn. This option trained the slowest and saw the worst results. This is expected and these kinds of nn are meant to have more robust testing and fine-tuning infrastructure build around them, the kind that I didn't have time to implement. More on that in the "Potential Improvement" section.

### Class Structure and Inheritance

There a numerous attributes that are important to a real-world MLOps pipeline - automation, reproducibility, scalability... but one key important attribute that is not talked about often enough is flexibility. We want our customers, the ML scientists, to be able to create various models, train, test and fine-tune them with easy with minimal boilerplate. That's why it's important to create a coding framework that uses inheritance to define a hierarchy of model architectures. This is a framework that I've successfully deployed before and replicate here in a simplified manner.

**RedditScorePredictorBase** - This is the base class for all predictors. It contains common functionality that doesn't change between architectures. It also acts as a template for specific implementation. I've deliberately not declared it as an abstract class.

**RedditScorePredictorSimpleBase** - This is the base class that implements most of the functionality for both options 1 and 2.

**RedditScorePredictorSimpleRandomForest** - Inherits from 'RedditScorePredictorSimpleBase', by only setting `self.model` to scikit-learn's RandomForestRegressor.

**RedditScorePredictorSimpleNN** - Same as above, but `self.model` is set to scikit-learn's MLPRegressor.

**RedditScorePredictorAdvanced** - This is the final architecture. It inherits directly from RedditScorePredictorBase.

### Functionality

**new_model** - class method - Use to create a new model from scratch. Calls the constructor.

**save_model** - class method - Use to save a model after training, so it can be loaded at any time from file and run an inference.

**load_model** - class method - Load a previously saved model.

**load_data** - static method - Use to load the dataset ([askscience_data.csv](askscience_data.csv))

**prepare_features** - instance method - Use to prepare features for training, testing or inference. Depending on the model architecture, it will do slightly different things, but in general it will do feature selection, text cleaning (e.g., remove stopwords, lowercase, etc.), handling missing tabular data and apply feature transformation.

**train** - instance method - This is the method that will do a single train. It assumes that the data is already split between training set and testing set. It only works on the training set. For the simpler scikit-learn's models (option 1 and 2), the training set is used as is for running a single training iteration. For the advanced model, however, the training set is further split between a proper training set and a validation set, which is used by the architecture to track performance metric improvements after each epoch.

**validate** - instance method - Only available to option 3. Runs validation after each training epoch. Not to be confused with 'test'.

**train_with_cross_validation** - instance method - This method is available only to option 1 and 2. Didn't have time to implement for option 3. It will run a train on the full dataset, after in has passed cross validation on X number of folds.

**test** - instance method - Used to run a single test iteration on the previously segregated test data set.

**predict_batch** - instance method - Run inference on a batch of new posts in the form of a dataframe.

**predict** - instance method - Run inference on a single post in the form of dict.

**training_pipeline** - class method - Point of entry into the training process.

### Modules

The python code is split into the following modules:  

[common.py](modules/common.py) - Contains code that is shared between the different models.

[option_1.py](modules/option_1.py) - Entry point for training option 1. It also contains its specific constructor. To run training, simply call from project's root `python modules/option_1.py`

[option_2.py](modules/option_2.py) - Entry point for training option 2. `python modules/option_2.py`

[option_3.py](modules/option_3.py) - Entry point for training option 3. `python modules/option_3.py`

[prediction.py](modules/prediction.py) - Entry point for making a prediction. You'll have to comment-in the predictor class you want to use. `python modules/prediction.py`. This assumes you have already trained the associated predictor model.

[memory.py](modules/memory.py) - Small utility for tracking memory usage.

### Discussion

**Most of the answers bellow apply generally to a real-world production system**

#### Discuss and reflect on the performance of your model. What metrics did you use to evaluate it? How would you improve it?

The performance of the model was evaluated based on Mean Squared Error (MSE) and Root Mean Squared Error (RMSE), as these metrics effectively quantify the magnitude of the error in regression tasks. MSE captures the average squared difference between predicted and true values, penalizing larger deviations. RMSE provides a more interpretable measure in the same unit as the target variable by taking the square root of MSE.

**Random Forest** performed reasonably well, with a balance between simplicity and accuracy. However, it struggled to generalize well and left some posts with huge gaps between predicted and actual scores.

**MLPRegressor** leveraged tabular features, but showed sensitivity to hyperparameters and sometimes overfit without careful regularization or early stopping.

**BERT-based neural network** (advanced architecture) I thought it would provide the best results for text-heavy data, given its ability to capture contextual meaning. However, training was computationally expensive, and performance was limited by dataset size and preprocessing power on my local mac. I had to constantly reduce overall data set size.

#### Discuss MLOps metrics that you would track in a production deployment of this model - model drift

Data Distribution Metrics: Compare the statistical properties (mean, variance, KL divergence, missing values) of incoming data against the training dataset.

Concept Drift Metrics: Monitor prediction accuracy on periodically scored posts.

Feature Importance Drift: Measure the change in the model's feature importance over time.

Use MLflow to log model performance over time and compare against benchmarks.

Prediction Distribution: Monitor if predicted scores cluster or diverge over time.

Use other custom dashboards like Prometheus and Grafana to display drift metrics.

Implement automated retraining pipelines triggered when drift exceeds a predefined threshold.

Use online learning or incremental updates for models sensitive to data distribution changes.

#### MLOps metrics - Inference Latency

Average Latency: Track the time to process a single post.

P95/P99 Latency: Measure latency at the 95th and 99th percentiles to identify outliers.

Throughput: Measure how many posts per second each model can handle.

We could cache scores for frequently queried posts.

#### MLOps metrics - Data Quality

Data quality directly impacts prediction accuracy.

Tracking null or missing values is useful. Test different imputation mechanisms and figure out which ones produce the best performance metrics.

Outlier Detection: Detect anomalies or unexpected feature values using z-scores or interquartile range (IQR) for the tabular data.

Automate data quality checks wherever we can.

Flag and route anomalous feature values for manual review or exclusion.

#### MLOps metrics - Computational Resources

Along with the other metrics, we should also track general resource utilization metrics - CPU/GPU utilization, memory utilization, disk I/O, etc., during training and inference. 

#### If you did use an LLM, describe its impact on model performance

I used an LLM in option 3 and for the same computational resources I used for option 1 and 2, the performance of the model actually decreased. However, I'm sure that if proper GPU resources where made available, and ample time, I expect the performance to increase.

For improvement, given infinite time and resources, see bellow.

### Potential Improvements

Currently most of the hyperparameters are clumped together with the rest of the code. This is not good. I had well and good intentions to actually create a proper hyperparameter framework, where the user would be able to define sets of hyperparameters and run the training pipeline for each such set. This is how we would find the best set of hyperparameters, and that's how it's done in production. 

All hyperparameters and any other model configurations would be best stored in a yaml file for easy manipulation.

If resources permit, we could also use multiprocessing to test different sets of hyperparameters simultaneously.

Further we could use a library like 'hyperopt' for efficiently traversing through hyperparameter combinatorial space. 

I could try adding 'author' as a feature. Maybe some authors are more popular than others.

Deploy in the cloud with Terraform and Terragrunt.

I'm currently not scaling the embeddings themselves. Only the tabular data. I could try that, also I could try scaling the target variable. 

I could try to use a more robust loss function - If the dataset has outliers or is highly imbalanced, the default squared loss might amplify errors. I can try Huber loss.

Make it so `tag` and `upvote_ratio` are optional during inference. Add logic to handle missing data for all options (1, 2, 3).

I have a bunch of print statement everywhere. Replace with logging in a real-world scenario. 

Last, but not least, I wanted to add tests, but definitely not enough time.

#### Option 3 Specific Improvements

I could have fine-tuned BERT on the dataset (though this requires even more computation).

Currently, I used the so-called [CLS] embedding token, which is a single embedding for each text as a whole, but I could use Token-Level  embeddings, and then use additionally an LSTMs, or some other sequence model architecture so we don't loose context. This should increase accuracy for large complicated texts, but not sure if will provide any benefits here, even if we had the time and resources to train.

Early stopping is build-in the scikit-learn's MLPRegressor, but I didn't get to implement it in option 3. I had full intentions to do so, but no time. I was also going to use a learning rate scheduler with `from torch.optim.lr_scheduler import StepLR`. 

### Notes

I started implementing a 4th version, which replaces the BERT LLM for the feature embeddings and uses BigBird instead. The reason being that BERT has a max token length of 512 tokens. BirBird has ~8k. There are post bodies that exceed 5k tokens. However, I was already struggling running training on BERT on my local mac. I can't imagine running BigBird LLM. This does mean that for many posts, most of the text is currently truncated. There are other methods that I could have used, like the pooled embeddings method for getting all tokens, or running a summarization LLM job first. 

I didn't get to implementing cross-validation for option 3, but I don't think it was going to be feasible anyway.
