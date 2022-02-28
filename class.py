import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from os.path import dirname
import numpy as np
import os 
import sys
import pandas as pd
import numpy as np

sys.path.append(dirname(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

class CNN(nn.Module):
    def __init__(self, bsz):
        super().__init__()
        self.bsz = bsz
        self.layer=nn.Sequential(
            
            nn.Conv2d(3,16,3, bias=False), # bias 제거
            nn.BatchNorm2d(16),
            nn.ReLU(),
            #nn.MaxPool2d(2,2),

            nn.Conv2d(16,32,2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.MaxPool2d(2,2),
            
            nn.Conv2d(32,64,2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(64,128,2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            #nn.Flatten()
            )

        self.fc_layer=nn.Sequential(
            nn.Linear(144*32,self.bsz, bias=False),   
            nn.BatchNorm1d(self.bsz),
            nn.Linear(self.bsz,10,bias=False),
            nn.Softmax()
            
        )

    def forward(self,x):
        out=self.layer(x)
        out=out.view(out.shape[0],-1)
        out=self.fc_layer(out)
        
        return out

class CNN_module:
    def __init__(self, bsz):
        self.bsz = bsz
    
    def load_train_data(self):
        self.columns = ['bird', 'cat', 'ship', 'deer', 'horse', 'dog', 'truck', 'airplane', 'frog', 'automobile']
        with open('train_data_x.pickle', 'rb') as f:
            train_data_x = pickle.load(f) 

        train_data_y = []
        for i in range(10):
            tmp = np.zeros((10))
            tmp[i] = 1
            for ii in range(5000):
                train_data_y.append(tmp)

        train_data_y = np.array(train_data_y)

        return train_data_x, train_data_y

    def load_model(self):
        self.seq = CNN(self.bsz)
        self.seq.to(device)
        
        self.optimizer = torch.optim.Adam(self.seq.parameters(), betas=(0.9, 0.98), eps=1e-09)

        self.train_criterion = nn.CrossEntropyLoss()
        self.eval_criterion = nn.MSELoss(reduction=False)

    def load_test_data(self):
        with open('test_data_x.pickle', 'rb') as f:
            test_data_x = pickle.load(f) 

        return test_data_x

    def train(self, input_seq, output_seq, epochs):
        self.seq.train()

        num_iters = len(input_seq) // self.bsz
        acc_loss = np.zeros((num_iters))
        idx = np.arange(len(input_seq))

        np.random.shuffle(idx)

        input_seq = np.transpose(input_seq, (0,3,1,2))

        for i in range(num_iters):

            self.optimizer.zero_grad()
            
            inputs = np.zeros((self.bsz,3,32,32))
            targets = np.zeros((self.bsz,10))
            for ii in range(self.bsz):
                inputs[ii,:,:,:] = input_seq[idx[i*self.bsz+ii],:,:,:]
                targets[ii,:] = output_seq[idx[i*self.bsz+ii],:]
            inputs = torch.FloatTensor((inputs)).cuda()
            targets = torch.FloatTensor((targets)).cuda()

            outputs = self.seq(inputs)

            loss = self.train_criterion(outputs, targets)

            acc_loss[i] = loss.detach().cpu().numpy()

            print('{}\tof {}\ttrain_loss {:.2e}\t at epoch {}'.format(i+1, num_iters, acc_loss[i], epochs))

            if torch.isfinite(loss):
                loss.backward()
                grads = [x.grad for x in self.seq.parameters()]
                if not torch.isnan(torch.sum(grads[0])):
                    torch.nn.utils.clip_grad_norm_(self.seq.parameters(), 0.1) # 미분값을 0.1 이하로 제한
                    self.optimizer.step()
            else:
                print('loss is infinite')
                exit(1)

        return acc_loss

    def eval(self, input_seq):
        self.seq.eval()

        num_iters = len(input_seq) // self.bsz
        acc_outputs = np.zeros((len(input_seq),10))
        idx = np.arange(len(input_seq))

        np.random.shuffle(idx)

        input_seq = np.transpose(input_seq, (0,3,1,2))
        #[N, C, W, H]

        for i in range(num_iters):
            
            inputs = np.zeros((self.bsz,3,32,32))
            for ii in range(self.bsz):
                inputs[ii,:,:,:] = input_seq[idx[i*self.bsz+ii],:,:,:]

            with torch.no_grad():
                inputs = torch.FloatTensor((inputs)).cuda()

                outputs = self.seq(inputs)
                #print(outputs[0], targets[0])

            acc_outputs[i*self.bsz:(i+1)*self.bsz,:] = outputs.detach().cpu().numpy()

            print('{}\t of {}'.format(i+1, num_iters))

        if len(idx) % self.bsz != 0:
            i += 1
            bsz = len(idx) % self.bsz
            inputs = np.zeros((bsz,3,32,32))
            for ii in range(bsz):
                inputs[ii,:,:,:] = input_seq[idx[i*self.bsz+ii],:,:,:]

            with torch.no_grad():
                inputs = torch.FloatTensor((inputs)).cuda()

                outputs = self.seq(inputs)
                #print(outputs[0], targets[0])

            acc_outputs[i*self.bsz:(i+1)*self.bsz,:] = outputs.detach().cpu().numpy()

            print('{}\t of {}'.format(i+1, num_iters))


        return acc_outputs

    def export(self, epoch, outputs):
        tmpoutput = []
        for output in outputs:
            tmpoutput.append(self.columns[int(np.where(output==max(output))[0])])
            
        #print(tmpoutput)
        result = pd.DataFrame(data={'class':tmpoutput})

        result.to_csv("epoch_{}_outputs.csv".format(epoch))


if __name__ == "__main__":
    num_epochs = 1000
    batch_size = 128

    cnn_module = CNN_module(batch_size)

    cnn_module.load_model()
    train_x, train_y = cnn_module.load_train_data()
    test_x = cnn_module.load_test_data()
    for epoch in range(1, num_epochs+1):
        print('start train epoch {} of {}'.format(epoch, num_epochs))

        train_loss = cnn_module.train(train_x, train_y, epoch)

        if epoch % 10 == 0:

            test_outputs = cnn_module.eval(test_x)

            cnn_module.export(epoch, test_outputs)

    

    