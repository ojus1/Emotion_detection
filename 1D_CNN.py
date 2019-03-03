from torch import nn
from torch.utils.data import DataLoader
import torch
from DataHelper import VideoFramesDataset
from sklearn.metrics import log_loss

def model_score(model, dataset):
    loss = log_loss(dataset[0][1], model(dataset[0][0]))
    for x, y in dataset:
        loss += log_loss(y, model(x))
    return loss / len(dataset)

def checkpoint(model, val_dataset, best_loss):
    val_loss = model_score(model, val_dataset)
    print("------------------------------------------------------------------------------")
    print("Validation loss:", val_loss, "Best loss:", best_loss)
    print("------------------------------------------------------------------------------")
    if val_loss > best_loss:
        print("Saving model, new best loss", val_loss)
        torch.save(model, "models/resnet_cnn_{}.pth".format(val_loss))
        return val_loss
    print("Not saving model")
    return None

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Video1DCNN(nn.Module):
    def __init__(self, batch_size):
        super(Video1DCNN, self).__init__()
        def ConvBlock(in_channels, out_channels, stride=2, padding=0, k=5):
            return [nn.Conv1d(in_channels, out_channels, k, stride=stride, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.MaxPool1d(2)]
        
        self.batch_size = batch_size

        self.norm1 = nn.BatchNorm3d(45,)
        self.conv_block = ConvBlock(45, 128, k=5, stride=2)
        self.conv_block += ConvBlock(128, 256, k=5, stride=1)
        self.conv_block += ConvBlock(256, 512, k=3, stride=1)
        self.conv_block += ConvBlock(512, 512, k=3, stride=1)
        self.conv_block += ConvBlock(512, 512, k=3, stride=1)
        self.conv_block += ConvBlock(512, 512, k=3, stride=1)
        self.conv_block = nn.ModuleList(self.conv_block)

        self.fc1 = nn.Linear(512, 128)
        self.norm2 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 9)

        self.lrelu = nn.LeakyReLU(0.01)
        
    def forward(self, x):
        #print(x.shape)
        out = x
        for layer in self.conv_block:
            #print(layer.__class__.__name__)
            if layer.__class__.__name__ == "Conv1d":
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
    batch_size = 2
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = VideoFramesDataset(root_dir="data/video/train/")
    val_dataset = VideoFramesDataset(root_dir="data/video/validation/")

    #Initialize the model with random weights
    model = Video1DCNN(batch_size)
    model.apply(weights_init_normal)

    #Setting device to do compute    
    if torch.cuda.is_available():
        model = model.cuda()
        loss_func = loss_func.cuda()
        Tensor = torch.cuda.FloatTensor
        dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=3, pin_memory=True)

    else:
        Tensor = torch.FloatTensor
        dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=0)
    
    #model = model.cuda()
        
    '''
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=1e-1,
                                momentum=0.9,
                                nesterov=True,
                                weight_decay=1e-3)
    '''
    optimizer = torch.optim.Adam(model.parameters())


    itr = 0
    best_loss = 10000
    avg_loss = 30
    for epoch in range(num_epochs) :
        for X, y in dataloader :
            optimizer.zero_grad()
            #X = X.cuda()
            #y = y.float().cuda()
            y = y.float()

            y_pred = model(X)

            loss = loss_func(y_pred, y)

            loss.backward()
            optimizer.step()
            
            avg_loss = (avg_loss + loss.item()) / 2
            
            chckpt = checkpoint(model, val_dataset, best_loss)
            if chckpt is not None:
                best_loss = chckpt

            print('epoch [{}/{}], loss:{:.4f}, avg_loss: {:.4f}, uIter: {}'.format(epoch+1, num_epochs, loss.item(), avg_loss, itr))
            itr += 1
            

if __name__ == "__main__":
    train_CNN()
    #model = Video3DCNN(2)
    #x = torch.rand(2, 3, 45, 224, 224)
    #out = model(x)
    #print(out)
