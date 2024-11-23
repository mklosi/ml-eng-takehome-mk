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
