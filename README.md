# Intent Detection using HINT3 V1 data

## Contents
- [Setup](#setup)
- [Model Details](#model-details)
- [Training](#training)
- [Results](#results)
- [Further Improvement](#further-improvement)

## Setup
The model requires transformers and pytorch. Run the following command to install the required packages in a virtual environment.
```bash
pip install -r requirements.txt
git clone https://github.com/hellohaptik/HINT3.git
```

## Model Details

## Training

## Results
### Train Results
- Loss: 0.277
- Accuracy: 0.996
- F1 Score: 0.996

### Valid Results
- Loss: 0.843
- Accuracy: 0.864
- F1 Score: 0.858

### Test Results
- Accuracy: 0.375
- F1 Score: 0.292

### Test Updated Results
- Accuracy: 0.645
- F1 Score: 0.631

Model has overfit to training data quite a bit. But it is also able to perform well on validation data. The bad results in the test are due to the domain shift that's happening. This could be solved by collecting more data.

## Further Improvement
- Increase the training data
- Unfreeze the DistillBERT model layers step by step
- Add a learning rate scheduler
- Do hyperparameter tuning
- Change the backend model - DistillBERT to RoBERTa or BERT.