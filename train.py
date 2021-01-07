import torch.nn.functional as F
import torch
from params import batch_size
import numpy as np

def my_train_model(num_layers, num_hidden_fc, num_hidden_embedding, device, dataset, model, optimizer, num_epochs, max_sequence_length):

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.1, verbose=True)
    least_valid_loss = 1e12
    patience_counter = 0
    patience_e = 11

    h = torch.zeros(num_layers, batch_size, num_hidden_fc).to(device)
    c = torch.zeros(num_layers, batch_size, num_hidden_fc).to(device)

    losses = []
    valid_loss_list = []
    for epoch in range(num_epochs):

        # training mode
        dataset.set_partition(dataset.train)
        model.train()
        total_train_loss = 0
        total_train_correct = 0
        count = 0
        for x, y in dataset.get_batches():
            # for every batch in the training dataset perform one update step of the optimizer.

            # pad inputs to be the same sequence length and batch size
            len_x = x.shape[1]
            x_padded = torch.zeros((batch_size, max_sequence_length), dtype=torch.int64)
            if x.shape[0] < batch_size:
                x_padded[:x.shape[0], :len_x] = x
                y_padded = torch.zeros((batch_size), dtype=torch.int64)
                y_padded[:x.shape[0]] = y
                y = y_padded.to(device)
            else:
                x_padded[:, :len_x] = x
            x = x_padded.to(device)

            model.zero_grad()
            y_h, h, c = model(x, h, c)
            loss = F.cross_entropy(y_h, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_train_correct += (y_h.argmax(-1) == y).float().mean()
            count += 1
        average_train_loss = total_train_loss / count
        average_train_accuracy = total_train_correct / count

        # validation mode
        dataset.set_partition(dataset.valid)
        model.eval()
        total_valid_loss = 0
        total_valid_correct = 0
        count = 0
        for x, y in dataset.get_batches():
            # pad inputs to be the same sequence length and batch size
            len_x = x.shape[1]
            x_padded = torch.zeros((batch_size, max_sequence_length), dtype=torch.int64)
            if x.shape[0] < batch_size:
                x_padded[:x.shape[0], :len_x] = x
                y_padded = torch.zeros((batch_size), dtype=torch.int64)
                y_padded[:x.shape[0]] = y
                y = y_padded.to(device)
            else:
                x_padded[:, :len_x] = x
            x = x_padded.to(device)

            y_h, h, c  = model(x, h, c )
            loss = F.cross_entropy(y_h, y)
            total_valid_loss += loss.item()
            total_valid_correct += (y_h.argmax(-1) == y).float().mean()
            count += 1
        average_valid_loss = total_valid_loss / count
        losses.append((average_train_loss, average_valid_loss))
        average_valid_accuracy = total_valid_correct / count

        print(
            f'epoch {epoch} accuracies: \t train: {average_train_accuracy}\t valid: {average_valid_accuracy}\t valid_loss: {average_valid_loss}')
        dataset.shuffle()

        scheduler.step(average_valid_loss)
        valid_loss_list.append(average_valid_loss)

        # if minimum number of epochs are trained
        if epoch > patience_e:
            least_valid_loss, patience_counter = early_stopper(valid_loss_list, least_valid_loss, patience_counter,
                                                               patience_e)

            # if early stopping condition meets, break training
            if patience_counter > 9:
                print(f'Early stopping')
                break

    # test mode
    dataset.set_partition(dataset.test)
    model.eval()
    total_test_correct = 0
    count = 0
    for x, y in dataset.get_batches():
        # pad inputs to be the same sequence length and batch size
        len_x = x.shape[1]
        x_padded = torch.zeros((batch_size, max_sequence_length), dtype=torch.int64)
        if x.shape[0] < batch_size:
            x_padded[:x.shape[0], :len_x] = x
            y_padded = torch.zeros((batch_size), dtype=torch.int64)
            y_padded[:x.shape[0]] = y
            y = y_padded.to(device)
        else:
            x_padded[:, :len_x] = x
        x = x_padded.to(device)

        y_h, h, c = model(x, h, c)
        total_test_correct += (y_h.argmax(-1) == y).float().mean()
        count += 1
    average_test_accuracy = total_test_correct / count

    return losses, (average_train_accuracy, average_valid_accuracy, average_test_accuracy)

# function for early stoppping 
def early_stopper(valid_loss_list, least_valid_loss, patience_counter, patience_e):
    valid_loss_list_cp = valid_loss_list.copy()
    valid_loss_list_cp = valid_loss_list_cp[::-1]

    current_valid_loss = valid_loss_list_cp[0]
    if current_valid_loss < least_valid_loss:
        least_valid_loss = current_valid_loss
        patience_counter = 0

    else:
        early_stop_condition = True
        for index in range(patience_e):
            if current_valid_loss < valid_loss_list_cp[index]:
                early_stop_condition = False
        if early_stop_condition is True:
            patience_counter += 1
        else:
            patience_counter = 0

    return least_valid_loss, patience_counter
