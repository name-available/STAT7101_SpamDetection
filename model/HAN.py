import torch
from torch import nn 
from torch.nn import functional as F
from torch import FloatTensor
from torch import LongTensor

LOW_SENTINEL = -100000000

class attention_layer(nn.Module):
    def __init__(self, feature_dim):
        
        super(attention_layer, self).__init__()
        self.feature_dim = feature_dim 
        self.W = torch.nn.Parameter(torch.zeros(feature_dim,feature_dim))
        self.b = torch.nn.Parameter(torch.zeros(feature_dim))
        torch.nn.init.normal_(self.W)
        torch.nn.init.normal_(self.b)
        self.context_vector = torch.nn.Parameter(
            data = torch.zeros([feature_dim])
        )
        torch.nn.init.normal_(self.context_vector, mean=0.0, std=1.0)
        return
    
    # ------------------------------------------------
    # Input has shape [batch_size, seq_len, features]
    # ------------------------------------------------
    def forward(self, x):
        global LOW_SENTINEL
        
        x1 = torch.tanh(torch.matmul(x, self.W) + self.b)
        x2 = torch.matmul( x1, self.context_vector)
        x2 = x2.reshape([x2.shape[0],x2.shape[1],1])
        x3 = F.softmax(x2,dim=1)
        x4 = x3 * x
        x5 = torch.sum(x4,dim=1,keepdims=False)
        return x5


# =================================================== #
# First layer of HAN
# =================================================== #
class word_encoder(nn.Module):
    def __init__(self, inp_emb_dim, hidden_dim, num_layers= 2):
        super(word_encoder,self).__init__()
        self.bi_gru = torch.nn.GRU(
            input_size = inp_emb_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = True,
            bidirectional = True
        )
        self.att = attention_layer(2*hidden_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        return
    
    
    def get_init_state(self, batch_size, device):
        num_directions = 2
        # h_0 should be of shape (num_directions * num_layers, batch_size, hidden_dim)
        h_0 = torch.zeros([num_directions * self.num_layers, batch_size, self.hidden_dim])
        return h_0
    
    # Input is a minibatch of sentences
    # X has shape [ batch_size, seq_len, features]
    def forward(self,x, h_0):
        gru_op = self.bi_gru(x, h_0)[0]
        att_op = self.att(gru_op)
        return att_op
    
        

# =================================================== #
# Second layer of HAN
# =================================================== #
class sentence_encoder(nn.Module):
    def __init__(self, inp_dim, hidden_dim, num_layers= 2):
        super(sentence_encoder,self).__init__()
        self.bi_gru = torch.nn.GRU(
            input_size = inp_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = True,
            bidirectional = True
        )
        self.att = attention_layer(2*hidden_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        return
    
    
    def get_init_state(self, batch_size, device):
        num_directions = 2
        # h_0 should be of shape (num_directions * num_layers, batch_size, hidden_dim)
        h_0 = torch.zeros([num_directions * self.num_layers, batch_size, self.hidden_dim])
        return h_0
    
    # Input is a minibatch of sentences
    # X has shape [ batch_size, seq_len, features]
    def forward(self, x, h_0):
        gru_op = self.bi_gru(x, h_0)[0]
        att_op = self.att(gru_op)
        return att_op
    

class HAN_op_layer(nn.Module):
    def __init__(self, inp_dimension, num_classes):
        super(HAN_op_layer,self).__init__()
        if num_classes == 2:
            num_classes = 1
        self.FC = nn.Linear(inp_dimension, num_classes)
        return 
    
    def forward(self,x):
        return self.FC(x)
    

class HAN(nn.Module):
    def __init__(self, inp_emb_dim = 384, hidden_dim = 128, num_classes = 2):
        super(HAN,self).__init__()
        self.word_enc = word_encoder(inp_emb_dim, hidden_dim)
        self.sent_enc = sentence_encoder(2*hidden_dim, hidden_dim)
        self.han_op = HAN_op_layer(2*hidden_dim, num_classes)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return
    
    def forward(self,x):
        h_0_word = self.word_enc.get_init_state(x.shape[0], device=self.device)
        word_enc_output = self.word_enc(x, h_0_word)
        h_0_sent = self.sent_enc.get_init_state(1, self.device)
        sent_enc_output = self.sent_enc(word_enc_output.unsqueeze(0), h_0_sent)
        output = self.han_op(sent_enc_output.squeeze(0))
        return output

def main():
    model = HAN()
    model.to('cuda')
    x = torch.randn([1,1,384]).to('cuda')
    y = model(x)
    print(y.shape)
    print(y)

if __name__ == '__main__':
    main()
