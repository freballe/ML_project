import torch
import csv              # to store results
from models import SiameseNoShare, SiameseShare, BaseNet
import torch.optim as optim
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from dlc_practical_prologue import generate_pair_sets
from helpers import train, test
from visualization import plot_results_acc, plot_results_loss
from statistics import mean, stdev
import argparse

print("NOTE: Select only one flag between --full and --lr_tuning. If no flag is used then the best model training will be executed.")
######################################################################

parser = argparse.ArgumentParser(description='NOTE: Select only one flag between --full and --lr_tuning. If no flag is used then the best model training will be executed.')

parser.add_argument('--full',
                    action='store_true', default=False,
                    help = 'Execute the full training on all models (default = False)')

parser.add_argument('--lr_tuning',
                    action='store_true', default=False,
                    help = 'Execute learning rate tuning, returns results of each model with different learning rates (default = False)')

args = parser.parse_args()
######################################################################

if args.full and args.lr_tuning:
    raise ValueError('Cannot have both --full and --lr_tuning')

# Global variables
img_size = 14
n_channel = 1
n_epochs = 150
batch_size = 32
n_rounds = 10
train_val_split = 0.8                 # training set percentage for train-validation split
torch.backends.cudnn.enabled = False
seed = 42
torch.manual_seed(seed)

if args.full:
    models_to_test = [BaseNet(), SiameseShare(), SiameseShare(), SiameseNoShare(), SiameseNoShare()]
    config_to_test = [{"aux_loss": False, "lr":0.005}, {"aux_loss": False, "lr":0.005},{"aux_loss": True, "lr":0.005},{"aux_loss": False, "lr":0.005},{"aux_loss": True, "lr":0.005}]
elif args.lr_tuning:
    learning_rates = [0.0005, 0.001, 0.005, 0.0075, 0.01, 0.05]
    models_to_test = []
    config_to_test = []
    # hyperparameters tuning
    for lr in learning_rates:
        models_to_test.append(BaseNet())
        config_to_test.append({"aux_loss": False, "lr":lr})
        models_to_test.append(SiameseNoShare())
        config_to_test.append({"aux_loss": True, "lr":lr})
        models_to_test.append(SiameseNoShare())
        config_to_test.append({"aux_loss": False, "lr":lr})
        models_to_test.append(SiameseShare())
        config_to_test.append({"aux_loss": True, "lr":lr})
        models_to_test.append(SiameseShare())
        config_to_test.append({"aux_loss": False, "lr":lr})
else:
    models_to_test = [SiameseNoShare()]
    config_to_test = [{"aux_loss": True, "lr":0.005}]
# init dictionary that store all the results
models_results = {}
for j in range(len(models_to_test)):
    models_results[str(j)] = {}

for j in range(len(models_to_test)):
    print("Loading data..")
    rounds_histories = []
    test_results = []
    for i in range(n_rounds):
        print("Round ", i)
        random_seed = i
        torch.manual_seed(random_seed)

        # Load data
        train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(1000)
        print("Number of target == 0 in training set: ", len([t for t in train_target if t == 0]))

        # Data standardization
        m = train_input.mean()
        s = train_input.std()
        train_input = (train_input - m)/s
        test_input = (test_input - m)/s         # use same mean and std for both training and testing standardization

        full_dataset = TensorDataset(train_input, train_classes, train_target)
        train_size = int(train_val_split * len(full_dataset))
        val_size = len(full_dataset) - train_size

        test_dataset = TensorDataset(test_input, test_classes, test_target)

        aux_loss = config_to_test[j]["aux_loss"]
        learning_rate = config_to_test[j]["lr"]

        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        data_loaders = {
            "train":train_loader,
            "val":val_loader
            }

        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Init NN
        model = models_to_test[j]
        
        # Print info about the model
        if i == 0:
            print(model)
            print("Total number of parameters: ", sum(p.numel() for p in model.parameters()))
            print("Number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

        # Define loss and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train
        model, history = train(model, optimizer, loss_fn, data_loaders, n_epochs, aux_loss=aux_loss)
        rounds_histories.append(history)
        
        # Test
        test_acc = test(model, test_loader)
        test_results.append(test_acc)
    models_results[str(j)]["histories"] = rounds_histories
    models_results[str(j)]["test_results"] = test_results

plot_results_acc(models_results)
plot_results_loss(models_results)

# write results to csv
with open('model_results.csv', mode='w') as results_file:
    file_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    file_writer.writerow(['model', 'aux_loss', 'mean', 'std_dev', 'lr'])

    for j in range(len(models_to_test)):
        test_results = models_results[str(j)]["test_results"]
        avg = mean(test_results) if len(test_results) > 1 else test_results[0]
        std = stdev(test_results) if len(test_results) > 1 else 0
        print("Model:", models_to_test[j].__class__.__name__ ,"(aux. loss = ", config_to_test[j]["aux_loss"],", lr = ", config_to_test[j]["lr"], "). Mean accuracy on test data: ", avg)
        print("Model:", models_to_test[j].__class__.__name__ ,". Standard deviation of accuracy on test data: ", std)
        file_writer.writerow([models_to_test[j].__class__.__name__, config_to_test[j]["aux_loss"], avg, std, config_to_test[j]["lr"]])
