'''
Train
'''

from __future__ import print_function
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from srcnn_data import get_training_set, get_test_set
from srcnn_model import SRCNN, VDSR
import os
import csv
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

#Settings
use_cuda = 1
upscale_factor = 3
batch_size = 10
test_batch_size = 10
learn_rate = 0.0001
epochs = 1000

seed = 3000
if use_cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(seed)
if use_cuda:

    torch.cuda.manual_seed(seed)


train_set = get_training_set(upscale_factor)
test_set = get_test_set(upscale_factor)
training_data_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, batch_size=test_batch_size, shuffle=False)


#srcnn = SRCNN()
vdsr = VDSR()
criterion = nn.MSELoss()

if(use_cuda):
    #srcnn.cuda()
    vdsr.cuda()
    criterion = criterion.cuda()

#optimizer = optim.Adam(srcnn.parameters(),lr=learn_rate)
optimizer = optim.Adam(vdsr.parameters(),lr=learn_rate)


def train(epoch, csvfile, csv_writer):
    epoch_loss = 0
    #srcnn.train()
    vdsr.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1])
        if use_cuda:
            input = input.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        input = torch.div(input, 255.0)
        #model_out = srcnn(input)
        model_out = vdsr(input)
        loss = criterion(model_out*255.0, target)
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.data))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    csv_writer.writerow([float(epoch_loss / len(training_data_loader))])
    csvfile.flush()


def test(csvfile, csv_writer):
    avg_psnr = 0
    #srcnn.eval()
    vdsr.eval()
    for batch in testing_data_loader:
        input, target = Variable(batch[0]), Variable(batch[1])
        if use_cuda:
            input = input.cuda()
            target = target.cuda()
        input = torch.div(input, 255.0)
        #prediction = srcnn(input)
        prediction = vdsr(input)
        prediction_255 = prediction * 255
        mse = criterion(prediction_255, target)
        psnr = 10 * log10(1 / mse.data)
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    csv_writer.writerow([float(avg_psnr / len(testing_data_loader))])
    csvfile.flush()


def checkpoint(epoch):
    if not os.path.exists("./checkpoint/"):
        os.makedirs("./checkpoint/")
    model_out_path = "./checkpoint/model_epoch_{}.pth".format(epoch)
    #torch.save(srcnn, model_out_path, )
    torch.save(vdsr, model_out_path, )
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == "__main__":
    # csv writer
    csvfile1, csvfile2 = open("ex3.csv", "a+"), open("ex4.csv", "a+")
    csv_writer1, csv_writer2 = csv.writer(csvfile1), csv.writer(csvfile2)
    #train_writer = SummaryWriter(log_dir='/private/workspace/server_usage_example/SRCNN/train')

    for epoch in range(1, epochs + 1):
        train(epoch, csvfile1, csv_writer1)
        test(csvfile2, csv_writer2)
        if(epoch%50==0):
            checkpoint(epoch)

