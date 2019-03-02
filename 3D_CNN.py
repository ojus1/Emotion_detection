from torch import nn
from torch.utils.data import DataLoader
import torch
from DataHelper import VideoFramesDataset


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Video3DCNN(nn.Module):
    def __init__(self, batch_size):
        super(Video3DCNN, self).__init__()
        def ConvBlock(in_channels, out_channels, stride=2, padding=2, k=5):
            return [nn.Conv3d(in_channels, out_channels, k, stride=stride, padding=padding),
                nn.BatchNorm3d(out_channels),
                nn.MaxPool3d(2)]
        
        self.batch_size = batch_size

        self.norm1 = nn.BatchNorm3d(3, )
        self.conv_block = ConvBlock(3, 8, k=5, stride=2)
        self.conv_block += ConvBlock(8, 16, k=5, stride=2)
        self.conv_block += ConvBlock(16, 32, k=3, stride=1)
        self.conv_block += ConvBlock(32, 64, k=3, stride=1)
        self.conv_block += ConvBlock(64, 128, k=3, stride=1)
        self.conv_block += ConvBlock(128, 256, k=3, stride=1)
        self.conv_block = nn.ModuleList(self.conv_block)

        self.fc1 = nn.Linear(1024, 256)
        self.norm2 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 9)

        self.lrelu = nn.LeakyReLU(0.01)
        
    def forward(self, x):
        #print(x.shape)
        out = x
        for layer in self.conv_block:
            #print(layer.__class__.__name__)
            if layer.__class__.__name__ == "Conv3d":
                out = self.lrelu(layer(out))
            else:
                out = layer(out)
        out = out.view(self.batch_size, -1)
        #print(out.shape)
        out = self.lrelu(self.fc1(out))
        out = self.norm2(out)
        out = self.fc2(out)
        return out


def train_CNN():

    loss_func = nn.BCEWithLogitsLoss()

    # Hyperparameters
    num_epochs = 50
    batch_size = 3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = VideoFramesDataset()
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=3)

    #Initialize the model with random weights
    model = Video3DCNN(batch_size)
    model.apply(weights_init_normal)

    if torch.cuda.is_available():
        model = model().cuda()
        loss_func = loss_func.cuda()
        Tensor = torch.cuda.FloatTensor
    else:
        Tensor = torch.FloatTensor

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=1e-1,
                                momentum=0.9,
                                nesterov=True,
                                weight_decay=1e-3)
    
    itr = 0
    best_loss = 10000
    for epoch in range(num_epochs) :
        for X, y in dataloader :
            #print(X.shape)
            optimizer.zero_grad()

            y_pred = model(Tensor(X))

            loss = loss_func(y_pred, y)

            loss.backward()
            optimizer.step()
            
            print('epoch [{}/{}], loss:{:.4f}, uIter: {}'.format(epoch+1, num_epochs, loss.item(), itr))
            itr += 1
            if best_loss > loss.item():
                best_loss = loss.item()
                torch.save(model, 'models/3DCNN_{}_{}.pt'.format(epoch, best_loss))

if __name__ == "__main__":
    train_CNN()
    #model = Video3DCNN(2)
    #x = torch.rand(2, 3, 20, 256, 256)
    #out = model(x)
    #print(out)