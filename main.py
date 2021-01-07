import os
from dataloader import LabelledTextDS
from model import *
from visualize import *
from train import my_train_model
import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dev = 'cuda' if torch.cuda.is_available() else 'cpu'  # If you have a GPU installed, use that, otherwise CPU
dataset = LabelledTextDS(os.path.join('data', 'labelled_movie_reviews.csv'), dev=dev)

# returns the maximum sequence length of the whole dataloader
def get_max_sequence_length(dataset):
    max_length = -1
    dataset.set_partition(dataset.train)
    for x, y in dataset.get_batches():
        current_length = x.shape[1]
        if current_length > max_length:
            max_length = current_length

    dataset.set_partition(dataset.valid)
    for x, y in dataset.get_batches():
        current_length = x.shape[1]
        if current_length > max_length:
            max_length = current_length

    dataset.set_partition(dataset.test)
    for x, y in dataset.get_batches():
        current_length = x.shape[1]
        if current_length > max_length:
            max_length = current_length

    return max_length

model_name = 'FastTextLSTM' # FCN, FCN_short, FastTextLSTM

# LSTM model
if model_name == 'FastTextLSTM':
    num_epochs = 10
    num_hidden_embedding = 500  # Number of hidden neurons in model
    num_hidden_fc = 128
    num_layers = 2
    max_sequence_length = get_max_sequence_length(dataset)
    # embeddings = torch.load(os.path.join('saved_models', 'word_embeddings.pth')).embeddings.weight.data
    model = FastTextLSTM(dev, num_hidden_fc, num_layers, max_sequence_length, len(dataset.token_to_id) + 2, num_hidden_embedding, len(dataset.class_to_id), word_embeddings=None).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    losses, accuracies = my_train_model(num_layers, num_hidden_fc, num_hidden_embedding, dev, dataset, model, optimizer,
                                        num_epochs, max_sequence_length)  # rnn

torch.save(model, os.path.join('saved_models', 'classifier.pth'))

print('')
print_accuracies(accuracies)
plot_losses(losses)
