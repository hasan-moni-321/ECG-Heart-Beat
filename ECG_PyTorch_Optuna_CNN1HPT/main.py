import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch 
from torch.utils.data import DataLoader
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam

import optuna 
import argparse
from ignite.engine import create_supervised_evaluator
from ignite.engine import create_supervised_trainer
from ignite.engine import Events
from ignite.metrics import Accuracy


import model_file 
import dataset_file  


train = pd.read_csv("/home/hasan/Data Set/heart beat classification/archive(2)/mitbih_train.csv", header=None)
test = pd.read_csv("/home/hasan/Data Set/heart beat classification/archive(2)/mitbih_test.csv", header=None)
train[187] = train[187].astype(int)
test[187] = test[187].astype(int)

# Fixing imbalance problem 
df0 = train[train[187]==0].sample(n=800, random_state=42)
df1 = train[train[187]==1].sample(n=800, random_state=42)
df2 = train[train[187]==2].sample(n=800, random_state=42)
df3 = train[train[187]==3].sample(n=641, random_state=42)
df4 = train[train[187]==4].sample(n=800, random_state=42)
train_new = pd.concat([df0, df1, df2, df3, df4], ignore_index=True)

# Shuffle datasets 
train_new = train_new.sample(frac=1).reset_index(drop=True)
test_new = test.sample(frac=1).reset_index(drop=True) 

# Features and Labels of train and test dataset
train_features = train_new.drop(187, axis=1)
train_label = train_new[187]
test_feature = test_new.drop(187, axis=1) 
test_label = test_new[187] 

# Train and Valid dataset
x_train, x_valid, y_train, y_valid = train_test_split(train_features, train_label, random_state=42, stratify=train_label) 

# Data Normalize and Standardize
scaler = MinMaxScaler()
normalized_xtrain = scaler.fit_transform(x_train)
std = StandardScaler()
standardized_xtrain = scaler.fit_transform(normalized_xtrain)

# Reshaping dataset for convolutional layer 
X_train = np.array(x_train).reshape(x_train.shape[0], x_train.shape[1], 1)
X_valid = np.array(x_valid).reshape(x_valid.shape[0], x_valid.shape[1], 1)
X_test = np.array(test_feature).reshape(test_feature.shape[0], test_feature.shape[1], 1)

Y_train = y_train.values
Y_valid = y_valid.values
Y_test = test_label.values

print(X_train.shape, X_valid.shape, X_test.shape, Y_train.shape, Y_valid.shape, Y_test.shape)


train_set = dataset_file.dataset(
                                 X_train, 
                                 Y_train
                                 )
valid_set = dataset_file.dataset(
                                 X_valid, 
                                 Y_valid
                                 )
test_set = dataset_file.dataset(
                                 X_test, 
                                 Y_test
                                 )

train_loader = DataLoader(
                          train_set,
                          batch_size=32,
                          shuffle=True,
)
valid_loader = DataLoader(
                          valid_set,
                          batch_size=32,
                          shuffle=False,
)
test_loader = DataLoader(
                         test_set,
                         batch_size=32,
                         shuffle=False, 
)  



def objective(trial):
    # Create a convolutional neural network.
    model = model_file.ecg_net(trial)

    device = "cpu"
    if  torch.cuda.is_available():
        device = "cuda"
        model.cuda(device)

    #optimizer = Adam(model.parameters())

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss() 

    trainer = create_supervised_trainer(model, optimizer, criterion,  device=device)            # F.nll_loss,
    evaluator = create_supervised_evaluator(model, metrics={"accuracy": Accuracy()}, device=device)

    # Register a pruning handler to the evaluator.
    pruning_handler = optuna.integration.PyTorchIgnitePruningHandler(trial, "accuracy", trainer)
    evaluator.add_event_handler(Events.COMPLETED, pruning_handler)

    # Load MNIST dataset.
    #train_loader, val_loader = get_data_loaders(TRAIN_BATCH_SIZE, VAL_BATCH_SIZE)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        evaluator.run(valid_loader)
        validation_acc = evaluator.state.metrics["accuracy"] 
        print("Epoch: {} Validation accuracy: {:.2f}".format(engine.state.epoch, validation_acc)) 

    trainer.run(train_loader, max_epochs=10)

    evaluator.run(valid_loader)

    #############################################
    # Model Saving and Loading
    #############################################
    # Model saving
    torch.save(model, "ECG_heart_beat_classification.pt")

    # Model loading
    model = torch.load("ECG_heart_beat_classification.pt") 

    #############################################


    return evaluator.state.metrics["accuracy"]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Ignite example.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    args = parser.parse_args()
    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=10, timeout=600)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

