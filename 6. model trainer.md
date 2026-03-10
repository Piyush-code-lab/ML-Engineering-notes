# Model Training Component (Model Trainer) – End-to-End ML Project

In an **end-to-end machine learning pipeline**, we have already completed:

1. **Data Ingestion** → Loading and splitting raw data
2. **Data Transformation** → Cleaning data, handling missing values, encoding features, scaling etc.

After transformation, we now have:

* `train_array`
* `test_array`
* `preprocessor.pkl`

Now we build the **Model Trainer Component**, whose job is:

* Train multiple ML models
* Evaluate them
* Select the best model
* Save the best model as a `.pkl` file

---
# Full model_trainer.py code
```python
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
            



            
        except Exception as e:
            raise CustomException(e,sys)
```
---
# Update utils.py
```python
import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
```
---
# Step 1 — Import Required Libraries

First we import all required modules and ML algorithms.

### Full Code

```python
import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
```

---

### Explanation

These imports serve the following purposes:

| Import            | Purpose                            |
| ----------------- | ---------------------------------- |
| `os`              | File and directory handling        |
| `sys`             | Exception handling support         |
| `dataclass`       | Creating configuration objects     |
| `sklearn models`  | Algorithms to train                |
| `r2_score`        | Evaluation metric                  |
| `CustomException` | Project-level error handling       |
| `logging`         | Logging important events           |
| `save_object`     | Utility function to save model     |
| `evaluate_model`  | Utility function to compare models |

---

### Why we import many algorithms?

When solving ML problems:

**We don't know beforehand which model performs best.**

So we test multiple models like:

* Linear Regression
* Random Forest
* Gradient Boosting
* KNN
* Decision Tree

Then select the **best performing model automatically**.

---

# Step 2 — Create Model Trainer Configuration

Each pipeline component has its own **configuration class**.

This stores paths or parameters required by the component.

---

### Full Code

```python
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
```

---

### Explanation

This class defines where the trained model will be saved.

Path created:

```
artifacts/model.pkl
```

---

### Why use a config class?

Advantages:

1. Centralized configuration
2. Easy modification
3. Cleaner architecture
4. Scalable pipeline

---

### What happens after this?

Later the **trained best model will be saved to this path**.

---

# Step 3 — Create ModelTrainer Class

Now we create the main class responsible for model training.

---

### Full Code

```python
class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
```

---

### Explanation

When the class is initialized:

```
self.model_trainer_config
```

stores the configuration object.

Through this object we can access:

```
self.model_trainer_config.trained_model_file_path
```

---

### Why do we do this?

So the class knows:

* where to save the trained model
* configuration values needed during execution

---

# Step 4 — Create the Model Training Function

This function starts the model training pipeline.

---

### Full Code

```python
def initiate_model_trainer(self, train_array, test_array):
```

---

### Inputs

| Parameter     | Description                  |
| ------------- | ---------------------------- |
| `train_array` | transformed training dataset |
| `test_array`  | transformed testing dataset  |

These arrays were returned by **Data Transformation component**.

Important structure:

```
features + target column
```

The **last column = target variable**.

---

# Step 5 — Add Exception Handling

We wrap code inside try-except.

---

### Code

```python
try:
    ...
except Exception as e:
    raise CustomException(e, sys)
```

---

### Why?

Because in production systems:

* Errors must be handled safely
* Logged properly
* Raised in standardized format

---

# Step 6 — Split Features and Target

Now we separate **features (X)** and **target (y)**.

---

### Full Code

```python
X_train = train_array[:, :-1]
y_train = train_array[:, -1]

X_test = test_array[:, :-1]
y_test = test_array[:, -1]
```

---

### Explanation

Dataset structure:

```
feature1 | feature2 | feature3 | target
```

Using slicing:

```
:-1  → all columns except last
-1   → last column
```

So:

| Variable  | Content           |
| --------- | ----------------- |
| `X_train` | training features |
| `y_train` | training target   |
| `X_test`  | test features     |
| `y_test`  | test target       |

---

### Why do we do this?

ML algorithms require:

```
model.fit(X_train, y_train)
```

So we must separate **inputs and outputs**.

---

# Step 7 — Create Dictionary of Models

We define the models we want to test.

---

### Full Code

```python
models = {
    "Linear Regression": LinearRegression(),
    "KNN": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "AdaBoost": AdaBoostRegressor()
}
```

---

### Explanation

This dictionary stores:

```
model name → model object
```

Example:

```
"Random Forest" → RandomForestRegressor()
```

---

### Why use a dictionary?

Because we can easily loop through models:

```
for model_name, model in models.items():
```

This allows automated evaluation.

---

# Step 8 — Evaluate Models

We now call the utility function `evaluate_model`.

---

### Code

```python
model_report = evaluate_model(
    X_train,
    y_train,
    X_test,
    y_test,
    models
)
```

---

### What this function does

Inside `evaluate_model`:

For each model:

1. Train model
2. Predict training data
3. Predict test data
4. Calculate R² score
5. Store results in dictionary

---

### Example output

```
{
 "Linear Regression": 0.81,
 "Random Forest": 0.87,
 "Decision Tree": 0.75
}
```

---

### Why do we do this?

To compare **multiple models automatically**.

---

# Step 9 — Find Best Model

Now we select the model with **highest score**.

---

### Code

```python
best_model_score = max(model_report.values())
best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

best_model = models[best_model_name]
```

---

### Explanation

Steps:

1. Get highest score
2. Find model name with that score
3. Retrieve the model object

---

### Example

```
Random Forest → 0.87
Linear Regression → 0.81
```

Best model:

```
Random Forest
```

---

# Step 10 — Check Minimum Performance

We add a safety check.

---

### Code

```python
if best_model_score < 0.6:
    raise CustomException("No best model found")
```

---

### Why?

If **all models perform badly**, something is wrong:

Possible issues:

* bad dataset
* incorrect features
* wrong preprocessing

So we stop the pipeline.

---

# Step 11 — Save Best Model

Now we save the best model.

---

### Code

```python
save_object(
    file_path=self.model_trainer_config.trained_model_file_path,
    obj=best_model
)
```

---

### What `save_object` does

It:

1. Creates directory if needed
2. Saves model using `pickle`

Final file created:

```
artifacts/model.pkl
```

---

### Why do we save the model?

Because later it will be used for:

* predictions
* deployment
* APIs
* production inference

---

# Step 12 — Calculate Final R² Score

We calculate test accuracy.

---

### Code

```python
predicted = best_model.predict(X_test)

r2_square = r2_score(y_test, predicted)

return r2_square
```

---

### What happens here

Steps:

1. Predict test data
2. Calculate R² score
3. Return performance metric

Example result:

```
R² = 0.87
```

Meaning:

```
Model explains 87% variance
```

---

# Step 13 — Testing the Model Trainer

After creating the component we run it.

---

### Code

```python
from src.components.model_trainer import ModelTrainer

model_trainer = ModelTrainer()

r2_score_value = model_trainer.initiate_model_trainer(
    train_array,
    test_array
)

print(r2_score_value)
```

---

### What happens

Execution flow:

```
Data Transformation
        ↓
Model Trainer
        ↓
Evaluate models
        ↓
Select best model
        ↓
Save model.pkl
        ↓
Return R² score
```

---

# Step 14 — Git Commit

After successful execution we push code.

Commands used:

```
git add .
git commit -m "Model Trainer"
git push origin main
```

---

# Final Output Generated

Inside **artifacts folder**

```
artifacts/
    model.pkl
    preprocessor.pkl
```

These are used later in **deployment**.

---

# Next Step in the Pipeline

Next component:

```
Model Deployment
```

Possible deployment methods:

* Docker
* AWS
* API service

---

# Step-Wise Summary (Very Important)

### Step 1

Import required libraries and ML algorithms.

---

### Step 2

Create `ModelTrainerConfig` class to store model path.

---

### Step 3

Create `ModelTrainer` class.

---

### Step 4

Initialize configuration inside constructor.

---

### Step 5

Create `initiate_model_trainer()` method.

---

### Step 6

Split transformed arrays into:

```
X_train
y_train
X_test
y_test
```

---

### Step 7

Define multiple machine learning models.

---

### Step 8

Evaluate models using `evaluate_model()` utility.

---

### Step 9

Find the best performing model.

---

### Step 10

Check if performance is acceptable.

---

### Step 11

Save best model using `pickle`.

---

### Step 12

Calculate final R² score.

---

### Step 13

Run and test the model trainer component.

---

### Step 14

Push code to GitHub.

---

✅ Final output:

```
artifacts/model.pkl
```

---

views and real projects.**
