import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def plot_losses(losses):
    plt.plot(losses)
    plt.legend(['Training', 'Validation'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, len(losses), 1))
    plt.show()


def plot_embeddings(embeddings, names, word_list=None, max_num_words=20):
    if word_list is not None:
        word_list = set(word_list)
        inds = [i for i in range(len(names)) if names[i] in word_list]
        embeddings = embeddings[inds]
        names = np.array(names)[inds]
    else:
        embeddings = embeddings[:max_num_words]
        names = names[:max_num_words]
    embeddings = PCA(2).fit_transform(embeddings.cpu())
    plt.scatter(embeddings[:, 0], embeddings[:, 1])
    for i in range(len(embeddings)):
        plt.annotate(names[i], embeddings[i])
    plt.show()


def print_accuracies(accuracies):
    print(f'training accuracy:\t\t{accuracies[0]*100:.3f} %')
    print(f'validation accuracy:\t{accuracies[1]*100:.3f} %')
    print(f'testing accuracy:\t\t{accuracies[2]*100:.3f} %')