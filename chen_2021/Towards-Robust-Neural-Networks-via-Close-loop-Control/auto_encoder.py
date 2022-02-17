import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
    
# Two-layer convolutional auto-encoder
class ConvAutoencoder(nn.Module):
    def __init__(self, channels):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], 4, stride=2, padding=1),
            nn.ELU(),
            nn.BatchNorm2d( channels[1]),
            nn.Conv2d(channels[1], channels[2], 4, stride=2, padding=1),
            nn.ELU(),
        )
        self.decoder = nn.Sequential(
 			nn.ConvTranspose2d(channels[2], channels[1], 4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(channels[1], channels[0], 4, stride=2, padding=1),
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Train auto-encoder for a given hidden layer index
# The training structure is designed for ResNet considered in this work
# Other net structure can be considered by replacing the forward propagation

def training(train_loader, test_loader, encoder, model, epoches, layer_index, lr, device=None):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 125], gamma=0.1)
    model.eval()
    encoder.train()
    total_step = len(train_loader)
    loss_history = 1000.
    for epoch in range(epoches):
        train_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                # Train auto-encoder for input data
                outputs = images
                # Train auto-encoder for the first hidden state
                if layer_index > 0:
                    outputs = F.relu(model.bn1(model.conv1(outputs)))
                # Train auto-encoder for the second hidden state
                if layer_index > 1:
                    outputs = model.layer1(outputs)
                # Train auto-encoder for the thrid hidden state
                if layer_index > 2:
                    outputs = model.layer2(outputs)
                # Train auto-encoder for the forth hidden state
                if layer_index > 3:
                    outputs = model.layer3(outputs)
            # Compute reconstruction loss
            reconstruction = encoder(outputs)
            loss = criterion(reconstruction, outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.data
        print ('Epoch [{}/{}], LR: [{}], Loss: {:.6f}'.format(epoch+1, epoches, lr_scheduler.get_last_lr(), train_loss/total_step))
        lr_scheduler.step()
        test_loss = testing(test_loader, encoder, model, layer_index, device)
        if test_loss < loss_history:
            loss_history = test_loss
            torch.save(encoder.state_dict(), 'Auto_encoder_{}.ckpt'.format(layer_index))

# Test the performance of given auto-encoder
def testing(test_loader, encoder, model, layer_index, device):
    criterion = nn.MSELoss()
    total_step = len(test_loader)
    model.eval()
    encoder.eval()
    test_loss = 0.
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            # Test auto-encoder for input data
            outputs = images
            # Test auto-encoder for the first hidden state
            if layer_index > 0:
                outputs = F.relu(model.bn1(model.conv1(outputs)))
            # Test auto-encoder for the second hidden state
            if layer_index > 1:
                outputs = model.layer1(outputs)
            # Test auto-encoder for the thrid hidden state
            if layer_index > 2:
                outputs = model.layer2(outputs)
            # Test auto-encoder for the forth hidden state
            if layer_index > 3:
                outputs = model.layer3(outputs)
            # Compute reconstruction loss
            reconstruction = encoder(outputs)  
            loss = criterion(reconstruction, outputs)
            test_loss += loss.data
    test_loss /= total_step
    print ('Testing loss is: {:.6f}'.format(test_loss))
    return test_loss
        
        
