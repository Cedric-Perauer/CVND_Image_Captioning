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
        super(DecoderRNN,self).__init__()
        self.hidden_size = hidden_size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(input_size = embed_size, 
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            dropout=0,
                            batch_first = True)
                             #hidden LSTM Layer
        
        self.linear = nn.Linear(hidden_size, vocab_size) 

        # initialize the hidden state (see code below)
        self.hidden = self.init_hidden()

        
    def init_hidden(self):
        ''' At the start of training, we need to initialize a hidden state;
           there will be none because the hidden state is formed based on perviously seen data.
           So, this function defines a hidden state with all zeroes and of a specified size.'''
        # The axes dimensions are (n_layers, batch_size, hidden_dim)
        return (torch.FloatTensor(1, 1, self.hidden_size).uniform_(0,1),
                torch.FloatTensor(1, 1, self.hidden_size).uniform_(0,1))
    
    def forward(self, features, captions):
        captions = captions[:,:-1]
        embed = self.word_embeddings(captions)
        feat = torch.cat((features.unsqueeze(1),embed),dim=1)
        out, hidden = self.lstm(feat)
        output = self.linear(out)
        return output   
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        '''
        BeamSearch: iteratively consider the set
        of the k best sentences up to time t as candidates to generate
        sentences of size t + 1, and keep only the resulting best k
        of them. This better approximates S = arg max S 0 p(S 0 |I).
        We used the BeamSearch approach in the following experi-
        ments, with a beam of size 20. Using a beam size of 1 (i.e.,
        greedy search) did degrade our results by 2 BLEU points on
        average.
        '''
        outs = []
        while(len(outs)!=max_len):
            out = self.linear(out.squeeze(dim=1))
            _ , idx = torch.max(out,1)
            outs.append(idx.cpu().numpy()[0].item())
            if(idx==1):
                break
            inputs = self.word_embeddings(idx).unsqueeze(1)
            
            
        return outs