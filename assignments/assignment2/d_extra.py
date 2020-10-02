from pathlib import Path
from typing import List, Optional, Dict
import pandas as pd
import numpy as np

"""
COMPETITION EXTRA POINTS!!
The below method should:
1. Handle any dataset (if you think worthwhile, you should do some preprocessing)
2. Generate a model based on the label_column and return the one with best score/accuracy

The label_column may indicate categorical column as label, numerical column as label or it can also be None
If categorical, run through these ML classifiers and return the one with highest accuracy: 
    DecisionTree, RandomForestClassifier, KNeighborsClassifier or a NaiveBayes
If numerical, run through these ML regressors and return the one with lowest R^2: 
    DecisionTree, RandomForestRegressor, KNeighborsRegressor or a Gaussian NaiveBayes
If None, run through at least 4 of the ML clustering algorithms in https://scikit-learn.org/stable/modules/clustering.html
and return the one with highest silhouette (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)

Optimize all parameters for the given score above. Feel free to choose any method you wish to optimize these parameters.
Write as a comment why and how you decide your model_type/parameter combination.
This method should not take more than 10 minutes to finish in a desktop Ryzen5 (or Core i5) CPU (no gpu acceleration).  

We will run your code with a separate dataset unknown to you. We will call the method more then once with said dataset, measuring
all scores listed above. The 5 best students of each score type will receive up to 5 points (first place->5 points, second->4, and so on).
Finally, the top 5 students overall (with most points in the end) will be awarded a prize!
"""


def generate_model(df: pd.DataFrame, label_column: Optional[str]) -> Dict:
    return dict(model=None, final_score=None)
