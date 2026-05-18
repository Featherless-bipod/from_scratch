import torch
import torch.nn as nn

from alphafold.MHA.MultiHeadAttention import MultiHeadAttention

class AttentionBlock(nn.Module):
    def __init__(self, hidden_size, intermediate_size, N_head):
        super().__init__()

        self.mha = MultiHeadAttention(hidden_size,hidden_size//N_head,N_head,-2,use_bias_for_embeddings=True)

        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.intermediate = nn.Sequential(
            nn.Linear(hidden_size,intermediate_size),
            nn.GELU(intermediate_size),
            nn.Linear(intermediate_size, hidden_size)
        )
        self.layer_norm_2 = nn.LayerNorm(hidden_size) 


    def forward(self, x, attention_mask=None):
        """
        Forward pass for AttentionBlock. The forward pass consists of the steps
        x -> mha -> h1 -> layer_norm(x+h1) -> intermediate -> h2 -> layer_norm(x+h2)

        Args:
            x (torch.tensor): Input tensor of shape (*, T, c) where T denotes the
                temporal dimension along which attention is performed, and c is the hidden_size.
            attention_mask (torch.tensor, optional): Attention mask of 
                shape (*, k). If not None, values that are equal to zero 
                in the mask are masked out during attention. Defaults to None.

        Returns:
            torch.tensor: Output tensor of shape (*, T, c).
        """

        out = None

        o = self.mha(x,attention_mask = attention_mask)
        o = self.layer_norm_1(o+x)
        o = self.intermediate(o)
        out = self.layer_norm_2(o+2)


        return out


class SentimentAnalysis(nn.Module):

    def __init__(self, vocab_size, hidden_size, intermediate_size, N_head, num_blocks, input_length):
        super().__init__()
        self.input_length = input_length

        self.word_embeddings = nn.Embedding(vocab_size,hidden_size,padding_idx=0)
        self.position_embeddings = nn.Embedding(input_length,hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.blocks = nn.ModuleList(
            [AttentionBlock(hidden_size,intermediate_size,N_head) for _ in range(num_blocks)] 
        )

        self.out = nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,2)
        )

    
    def forward(self, inp, attention_mask=None):
        out = None

        pos = torch.arange(inp.shape[-1],device=inp.device) % self.input_length
        embedding = self.word_embeddings(inp) + self.position_embeddings(inp)
        o = self.layer_norm(embedding)
        for block in self.blocks:
            o = block(o,attention_mask=attention_mask)

        out = self.out(o[...,0,:]) #the first token [CLS] is treated as a summary token
         

        return out

class SentimentWrapper(nn.Module):
    
    def __init__(self, model, learning_rate=2e-5):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, inp, attention_mask=None):
        out = None
        out = self.model(inp,attention_mask=attention_mask)
        return out

    def training_step(self, batch, batch_idx):
        inp, attn_mask, labels = batch['input_ids'], batch['attention_mask'], batch['label']

        out, loss, accuracy = None, None, None

        out = self(inp,attention_mask=attn_mask)
        loss = self.criterion(out,loss)
        accuracy = (out.argmax(dim=-1) == loss).float().mean()

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', accuracy, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):

        inp, attn_mask,labels = batch['input_ids'],batch['atttention_mask'],batch['label']
        out = self(inp, attention_mask = attn_mask)
        loss = self.criterion(out, labels)
        accuracy = (out.argmax(dim=-1) == labels).float().mean()

        self.log('val_loss',loss,prog_bar = True)
        self.log('val_accuracy',accuracy, prog_bar = True)

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(),lr=self.learning_rate)
    
        return optimizer

        
def map_keynames_from_distilbert(named_parameters):
    name_map = {
        'distilbert.': '',
        'embeddings.LayerNorm': 'layer_norm',
        'embeddings.position_embeddings': 'position_embeddings',
        'embeddings.word_embeddings': 'word_embeddings',
        'transformer.layer.': 'blocks.',
        'attention.': 'mha.',
        'q_lin.': 'linear_q.',
        'k_lin.': 'linear_k.',
        'v_lin.': 'linear_v.',
        'out_lin.': 'linear_o.',
        'sa_layer_norm': 'layer_norm_1',
        'ffn.lin1': 'intermediate.0',
        'ffn.lin2': 'intermediate.2',
        'output_layer_norm': 'layer_norm_2',
        'pre_classifier': 'out.0',
        'classifier': 'out.2',
        
    }

    new_parameters = dict()
    if isinstance(named_parameters, dict):
        named_parameters = named_parameters.items()

    for i, (key, value) in enumerate(named_parameters):
        for original, new in name_map.items():
            key = key.replace(original, new)
        new_parameters[key] = value

    return new_parameters
