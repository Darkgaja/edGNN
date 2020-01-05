import time
import numpy as np
import dgl
import torch
from torch.utils.data import DataLoader
import random

from utils.early_stopping import EarlyStopping
from utils.io import load_checkpoint, save_checkpoint

from core.data.constants import LABELS, TRAIN_MASK, TEST_MASK, VAL_MASK, GRAPH
from core.models.constants import NODE_CLASSIFICATION, GRAPH_CLASSIFICATION
from core.models.model import Model
from core.data.constants import GRAPH, N_RELS, N_CLASSES, N_ENTITIES
from core.models.F1Loss import F1_Loss


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels).cuda() if labels[0].is_cuda else torch.tensor(labels)


class App:

    def __init__(self, early_stopping=True):
        if early_stopping:
            self.early_stopping = EarlyStopping(patience=100, verbose=True)

    def train(self, data, model_config, learning_config, save_path='', mode=NODE_CLASSIFICATION):

        #loss_fcn = F1_Loss()
        loss_fcn = torch.nn.CrossEntropyLoss()

        labels = data[LABELS]
        # initialize graph
        if mode == NODE_CLASSIFICATION:
            train_mask = data[TRAIN_MASK]
            val_mask = data[VAL_MASK]
            dur = []

            # create GNN model
            self.model = Model(g=data[GRAPH],
                               config_params=model_config,
                               n_classes=data[N_CLASSES],
                               n_rels=data[N_RELS] if N_RELS in data else None,
                               n_entities=data[N_ENTITIES] if N_ENTITIES in data else None,
                               is_cuda=learning_config['cuda'],
                               mode=mode)

            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=learning_config['lr'],
                                         weight_decay=learning_config['weight_decay'])

            for epoch in range(learning_config['n_epochs']):
                self.model.train()
                if epoch >= 3:
                    t0 = time.time()
                # forward
                logits = self.model(None)
                loss = loss_fcn(logits[train_mask], labels[train_mask])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epoch >= 3:
                    dur.append(time.time() - t0)

                val_acc, val_loss = self.model.eval_node_classification(labels, val_mask)
                print("Epoch {:05d} | Time(s) {:.4f} | Train loss {:.4f} | Val accuracy {:.4f} | "
                      "Val loss {:.4f}".format(epoch,
                                               np.mean(dur),
                                               loss.item(),
                                               val_acc,
                                               val_loss))

                self.early_stopping(val_loss, self.model, save_path)

                if self.early_stopping.early_stop:
                    print("Early stopping")
                    break

        elif mode == GRAPH_CLASSIFICATION:
            K = 3
            self.accuracies = np.zeros(K)
            self.recall = np.zeros(K)
            self.precision = np.zeros(K)
            graphs = data[GRAPH]                 # load all the graphs

            # debug purposes: reshuffle all the data before the splitting
            random_indices = list(range(len(graphs)))
            random.shuffle(random_indices)
            graphs = [graphs[i] for i in random_indices]
            labels = labels[random_indices]

            # Create holdout set
            TESTSIZE = int(0.3 * len(graphs))
            test_graphs = graphs[:TESTSIZE]
            test_labels = labels[:TESTSIZE]
            graphs = graphs[TESTSIZE:]
            labels = labels[TESTSIZE:]

            model_dict = dict()
            
            for k in range(K): # K-fold cross validation

                # create GNN model
                self.model = Model(g=data[GRAPH],
                                   config_params=model_config,
                                   n_classes=data[N_CLASSES],
                                   n_rels=data[N_RELS] if N_RELS in data else None,
                                   n_entities=data[N_ENTITIES] if N_ENTITIES in data else None,
                                   is_cuda=learning_config['cuda'],
                                   mode=mode)

                optimizer = torch.optim.Adam(self.model.parameters(),
                                             lr=learning_config['lr'],
                                             weight_decay=learning_config['weight_decay'])

                if learning_config['cuda']:
                    self.model.cuda()

                print('\n\n\nProcess new k')
                start = int(len(graphs)/K) * k
                end = int(len(graphs)/K) * (k+1)

                print(str(start) + ":" + str(end))

                # testing batch
                testing_graphs = graphs[start:end]
                self.testing_labels = labels[start:end]
                self.testing_batch = dgl.batch(testing_graphs)

                # training batch
                training_graphs = graphs[:start] + graphs[end:]
                
                training_labels = labels[list(range(0, start)) + list(range(end+1, len(graphs)))]

                training_samples = list(map(list, zip(training_graphs, training_labels)))
                training_batches = DataLoader(training_samples,
                                              batch_size=learning_config['batch_size'],
                                              shuffle=True,
                                              collate_fn=collate)

                dur = []
                for epoch in range(learning_config['n_epochs']):
                    self.model.train()
                    if epoch >= 3:
                        t0 = time.time()
                    losses = []
                    training_accuracies = []
                    for iter, (bg, label) in enumerate(training_batches):
                        logits = self.model(bg)
                        loss = loss_fcn(logits, label)
                        losses.append(loss.item())
                        _, indices = torch.max(logits, dim=1)
                        correct = torch.sum(indices == label)

                        training_accuracies.append(correct.item() * 1.0 / len(label))
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    if epoch >= 3:
                        dur.append(time.time() - t0)
                    val_acc, val_precision, val_recall, val_loss = self.model.eval_graph_classification(self.testing_labels, self.testing_batch)
                    print("Epoch {:05d} | Time(s) {:.4f} | Train acc {:.4f} | Train loss {:.4f} "
                          "| Val accuracy {:.4f} | Val precision {:.4f} | Val recall {:.4f} | Val loss {:.4f}".format(epoch,
                                                                           np.mean(dur) if dur else 0,
                                                                           np.mean(training_accuracies),
                                                                           np.mean(losses),
                                                                           val_acc,
                                                                           val_precision,
                                                                           val_recall,
                                                                           val_loss))

                    is_better = self.early_stopping(val_loss, self.model, save_path)
                    if is_better:
                        self.accuracies[k] = val_acc
                        self.recall[k] = val_recall
                        self.precision[k] = val_precision
                        model_dict[val_loss] = self.model

                    if self.early_stopping.early_stop:
                        print("Early stopping")
                        break
                self.early_stopping.reset()
            
            # Load best performing model
            self.model = model_dict[min(model_dict.keys())]
            save_checkpoint(self.model, save_path)

            # Starting test on holdout set
            data[LABELS] = test_labels
            data[GRAPH] = test_graphs
            self.validate(data, None)
            
        else:
            raise RuntimeError

    def validate(self, data, load_path, mode=GRAPH_CLASSIFICATION):

        if (load_path != None):
            try:
                print('*** Load pre-trained model ***')
                self.model = load_checkpoint(self.model, load_path)
            except ValueError as e:
                print('Error while loading the model.', e)
                return

        print('*** Start testing***\n')

        if mode == GRAPH_CLASSIFICATION:
            labels = data[LABELS]
            size = labels.numpy().size
            class_size = size - labels.numpy().sum() 
            print('Loaded {0} graphs (Class:{1}) for validation'.format(size, class_size))
            graphs = data[GRAPH]
            batches = dgl.batch(graphs)

            acc, precision, recall, _ = self.model.eval_graph_classification(labels, batches)
        else:
            return

        print("\nTest Accuracy {:.4f}".format(acc))
        print("Test Precision {:.4f}".format(precision))
        print("Test Recall {:.4f}".format(recall))
        if recall == 0 and precision == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        print("Test F1 Score {:.4f}".format(f1))

    def test(self, data, load_path='', mode=NODE_CLASSIFICATION):

        if mode == NODE_CLASSIFICATION:
            try:
                print('*** Load pre-trained model ***')
                self.model = load_checkpoint(self.model, load_path)
            except ValueError as e:
                print('Error while loading the model.', e)

            test_mask = data[TEST_MASK]
            labels = data[LABELS]
            acc, _ = self.model.eval_node_classification(labels, test_mask)
        else:
            acc = np.mean(self.accuracies)
            recall = np.mean(self.recall)
            precision = np.mean(self.precision)

        print("\nMean cross validation testing results:")

        print("\nTest Accuracy {:.4f}".format(acc))
        print("Test Precision {:.4f}".format(precision))
        print("Test Recall {:.4f}".format(recall))
        if recall == 0:
            f1 = 0
        else:
            f1 = 2 * (recall * precision) / (recall + precision)
        print("Test F1 Score {:.4f}".format(f1))

        return acc
