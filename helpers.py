import torch
import torchvision
import copy
from torch import Tensor, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from matplotlib import pyplot as plt

def train(model, optimizer, loss_fun, data_loaders, epochs, aux_loss):
    """
    Train network given as parameter 
    """
    #print('train() called: model=%s, opt=%s(lr=%f), epochs=%d\n' % (type(model).__name__, type(optimizer).__name__, optimizer.param_groups[0]['lr'], epochs))
    
    history = {}
    history['val_acc'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['loss'] = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):         # loop over the dataset multiple times
        for phase in ['train','val']:
            if phase == 'train':
                model.train()           # Set model to training mode
            else:
                model.eval()            # Set model to validation mode
            
            epoch_loss         = 0.0
            num_train_correct  = 0
            num_train_examples = 0
            
            for i, batch in enumerate(data_loaders[phase], 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels, targets = batch
            
                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        if aux_loss:                        # only in training we can use the aux functions, can't use them in validation!
                            outputs, aux1_outputs, aux2_outputs = model(inputs)
                            loss1 = loss_fun(outputs, targets)
                            loss2 = loss_fun(aux1_outputs, labels[:,0])
                            loss3 = loss_fun(aux2_outputs, labels[:,1])
                            if i == 0 and epoch == epochs - 1 and False:
                                print("targets", targets)
                                print("output", torch.max(outputs, 1)[1])
                                print("labels", labels)
                                print("aux1", torch.max(aux1_outputs, 1)[1])
                                print("aux2", torch.max(aux2_outputs, 1)[1])
                            loss = loss1 + loss2 + loss3
                            epoch_loss += loss1.item()      # need to consider only the loss on targets (and not aux losses) when comparing the results with the validation ones    
                        else:
                            outputs = model(inputs)
                            if type(outputs) == tuple:
                                outputs = outputs[0]
                            loss = loss_fun(outputs, targets)
                            epoch_loss += loss.item()
                    else:
                        outputs = model(inputs)
                        if type(outputs) == tuple:
                                outputs = outputs[0]
                        loss = loss_fun(outputs, targets)
                        epoch_loss += loss.item()

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                                
                # for accuracy
                predicted_classes = torch.max(outputs, 1)[1]
                num_train_correct  += (predicted_classes == targets).sum().item()
                num_train_examples += inputs.shape[0]

            epoch_acc  = num_train_correct / num_train_examples
            epoch_loss = epoch_loss / len(data_loaders[phase].dataset)

            # print statistics
            if phase == 'val':
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)
            
                if epoch_acc >= best_acc:       # save weights with best validation accuracy
                    print("Saving weights")
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_acc = epoch_acc
            else:
                history['loss'].append(epoch_loss)
                history['acc'].append(epoch_acc)
            print('[Epoch %d/%d] phase: %s \t loss: %.4f acc: %.4f' % (epoch + 1, epochs, phase, epoch_loss, epoch_acc))


    # load weights with best validation accuracy
    model.load_state_dict(best_model_wts)
    print('Finished training.')
    return model, history

def test(net, test_loader):
    """
    Compute accuracy of predictions on test data.
    """
    print("Testing the model.")
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels, targets = data
            outputs = net(images)
            if type(outputs) == tuple: # case when we use aux losses
                outputs = outputs[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    print('Accuracy of the network on the test images: %d %%' % (accuracy))
    
    return accuracy
