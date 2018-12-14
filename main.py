import os
import utils
import torch
from torch.utils.data import DataLoader
from build_dataset import data_split, EMDataset
import torch.nn as nn
import model
import argparse
from torch.autograd import Variable
import torch.optim as optim
from evaluate import accuracy
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments', help="Directory containing params.json")
parser.add_argument('--image_dir', default='Image', help="Directory containing images")
parser.add_argument('--label_dir', default='label', help="Directory containing labels")
parser.add_argument('--checkpoints_dir', default='checkpoints_val_kernel_5')



def main():
    # Load the parameters from json file
    args = parser.parse_args()
    image_dir = args.image_dir
    label_dir = args.label_dir
    cp_dir = args.checkpoints_dir
    logging.basicConfig(filename='train_kernel5.log', level=logging.INFO)
    logging.info('Started')
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    #split data
    train_image, train_label, dev_image, dev_label, test_image, test_label = data_split(image_dir, label_dir)
    #load data
    train_dataset = EMDataset(train_image, train_label)
    dev_dataset = EMDataset(dev_image, dev_label)
    test_dataset = EMDataset(test_image, test_label)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=params.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset,
                                  batch_size=params.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset,
                                  batch_size=params.batch_size, shuffle=True)

    #use GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = model.Net(1,6)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(device)

    # define loss
    class_weights = torch.FloatTensor(params.class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()

    #initialize optimiser
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    #training and evaluate
    epoch_train_loss= []
    epoch_test_loss = []
    for epoch in range(params.num_epochs):

        net.train()
        logging.info('epoch'+ str(epoch+1)+' start!')
        f_train_acc = [[] for i in range(6)]
        f_test_acc = [[] for i in range(6)]
        train(net, train_dataloader, device, optimizer, criterion, epoch, epoch_train_loss,f_train_acc)
        logging.info('epoch' + str(epoch + 1) + ' train_loss: ' + str(epoch_train_loss))
        logging.info('epoch' + str(epoch + 1) + ' train_acc: ' + str(f_train_acc))
        #save model
        torch.save(net.state_dict(), os.path.join(cp_dir, 'cp{}.pth'.format(epoch + 1)))
        #evaluate the model on test set.
        test(net, test_dataloader, device, criterion, epoch, epoch_test_loss, f_test_acc)
        logging.info('epoch' + str(epoch + 1) + ' test_loss: ' + str(epoch_test_loss))
        logging.info('epoch' + str(epoch + 1) + ' test_acc: ' + str(f_test_acc))
        logging.info('epoch'+ str(epoch+1)+' end!')

def train(net, train_dataloader, device, optimizer, criterion, epoch, epoch_train_loss, f_train_acc):

    running_loss = 0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs
        train_batch, label_batch = data
        train_batch = train_batch.to(device)
        label_batch = label_batch.to(device)

        train_batch, label_batch = Variable(train_batch), Variable(label_batch)
        m, h ,w = train_batch.shape
        train_batch = train_batch.view(m,1,h,w)
        label_batch = label_batch.type(torch.long)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(train_batch)
        loss = criterion(outputs, label_batch)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        if i % 100 == 0:  # print every 100 iterations.
            train_acc = accuracy(outputs, label_batch)
            for j in range(6):
                f_train_acc[j].append(train_acc[j])

    running_loss = running_loss/len(train_dataloader)
    epoch_train_loss.append(running_loss)


def test(net, test_dataloader, device, criterion, epoch, epoch_test_loss, f_test_acc):
    running_loss = 0
    net.eval()
    for i, data in enumerate(test_dataloader, 0):
        # get the inputs
        test_batch, label_batch = data
        test_batch = test_batch.to(device)
        label_batch = label_batch.to(device)

        test_batch, label_batch = Variable(test_batch), Variable(label_batch)
        m, h, w = test_batch.shape
        test_batch = test_batch.view(m, 1, h, w)
        label_batch = label_batch.type(torch.long)

        # forward
        outputs = net(test_batch)
        loss = criterion(outputs, label_batch)
        running_loss += loss.item()

        # print statistics
        if i % 100 == 0:  # print every 100 iterations
            test_acc = accuracy(outputs, label_batch)
            for j in range(6):
                f_test_acc[j].append(test_acc[j])

    running_loss = running_loss/len(test_dataloader)
    epoch_test_loss.append(running_loss)


if __name__ == '__main__':
    main()
