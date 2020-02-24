import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(input_size = embed_size, hidden_size = hidden_size,
                            num_layers = num_layers,Dropout=0.1, batch_first = True) #hidden LSTM Layer
        
        self.linear = nn.Linear(hidden_size, vocab_size) 

        # initialize the hidden state (see code below)
        self.hidden = self.init_hidden()

        
    def init_hidden(self):
        ''' At the start of training, we need to initialize a hidden state;
           there will be none because the hidden state is formed based on perviously seen data.
           So, this function defines a hidden state with all zeroes and of a specified size.'''
        # The axes dimensions are (n_layers, batch_size, hidden_dim)
        return (torch.FloatTensor(1, 1, self.hidden_dim).uniform_(0,1),
                torch.FloatTensor(1, 1, self.hidden_dim).uniform_(0,1))
    
    def forward(self, features, captions):
        captions = captions[:,-1]
        embed = self.word_embeddings(captions)
        features = torch.cat((features.unsqueeze(1),captions),dim=1)
        out, hidden = self.lstm(features)
        output = nn.Linear(out)

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass