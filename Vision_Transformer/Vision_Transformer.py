# -*- coding: utf-8 -*-
"""Vision_Transformer.py from scratch

Visual Transformers by Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale."https://arxiv.org/pdf/2010.11929.pdf.

Transformers have been studied in the context of sequence-to-sequence modelling applications like natural language processing (NLP).
Their superior performance to LSTM-based Recurrent neural network gained them a powerful reputation, thanks to their ability to model long sequences.
A couple of years ago, transformers have been adapted to the [visual domain](https://arxiv.org/abs/2010.11929) and suprisingly demonstrated better performance compared to the long standing convolutional neural networks conditioned to large-scale datasets.
Thanks to their ability to capture global semantic relationships in an image, unlike, CNNs which capture local information within the vicinty of the convolutional kernel window.

"""

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import cv2
import math

device = torch.device("cpu")

import torch.nn as nn

class LightViT(nn.Module):
    def __init__(self, image_dim, n_patches=7, n_encoder_blocks=2, d=16, n_heads=8, num_classes=10):
        super(LightViT, self).__init__()

        self.image_dim = image_dim
        self.n_patches = n_patches
        self.n_encoder_blocks = n_encoder_blocks
        self.d = d
        self.n_heads = n_heads
        self.num_classes = num_classes

        ## Class Members

        ## 1B) Linear Mapping
        self.linear_map = LinearMapping(self.image_dim, self.n_patches, self.d)

        ## 2A) Learnable Parameter
        self.cls_token = nn.Parameter(torch.randn(1,1, self.d))

        ## 2B) Positional embedding
        # Function used below directly, See below

        ## 3) Encoder blocks
        self.encoder = []
        for i in range(self.n_encoder_blocks):
            self.temp_encoder = ViTEncoder(self.d, self.n_heads)
            self.encoder.append(self.temp_encoder)

        # 5) Classification Head
        self.classifier = (nn.Linear(self.d, self.num_classes))
    
    def forward(self, images):
        ## Extract patches
        patches_extract = patches(images, self.n_patches)

        ## Linear mapping
        out = self.linear_map(patches_extract)

        ## Add classification token
        b, _, _ = out.shape
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        out = torch.cat([cls_tokens, out], dim=1)

        ## Add positional embeddings
        pos_embeddings = get_pos_embeddings(self.d, out)
        out = out + pos_embeddings

        ## Pass through encoder
        for i in range(self.n_encoder_blocks):
            out =  self.encoder[i](out)

        ## Get classification token
        #out = reduce('b n e -> b e', reduction='mean')
        out = out[:, 0:1, :]
        out = out.squeeze()

        ## Pass through classifier
        out = self.classifier(out)

        return out

"""## 1. Image Patches and Linear Mapping

### A) Image Patches
Transfomers were initially created to process sequential data. In case of images, a sequence can be created through extracting patches. 
To do so, a crop window should be used with a defined window height and width. 
The dimension of data is originally in the format of *(B,C,H,W)*, when transorfmed into patches and then flattened we get *(B, PxP, (HxC/P)x(WxC/P))*, where *B* is the batch size and *PxP* is total number of patches in an image. 
In this example, one can set P=7. 

*Output*: A function that extracts image patches. The output format should have a shape of (B,49,16). The function will be used inside *LightViT* class.
"""

def patches(images, n_patches):
    patches = rearrange(images, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=images[0].shape[1]//n_patches, s2=images[0].shape[2]//n_patches)
    return patches

"""### B) Linear Mapping

Afterwards, the input are mapped using a linear layer to an output with dimension *d* i.e. *(B, PxP, (HxC/P)x(WxC/P))* &rarr; *(B, PxP, d)*. 
The variable d can be freely chosen, however, we set here to 8. 

*Output*: A linear layer should be added inside *LightViT* class with the correct input and output dimensions, the output from the linear layer should have a dimension of (B,49,8). 
"""

class LinearMapping(nn.Module):
    
    def __init__(self, image_dim, n_patches, emb_size):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear((image_dim[2]//n_patches) ** 2, emb_size))
                
    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x

"""## 2. Classifier Token and Positional embeddings

### A) Classifier Token

Beside the image patches, also known as tokens, an additional special token is appended to the the input to capture desired information about other tokens to learn the task at hand. 
Later on, this token will be used as input to the classifier to determine the class of the input image. 
To add the token to the input is equivilant to concatentating a learnable parameter with a vector of the same dimension *d* to the image tokens. 

*Output* A randomly initialised learnable parameter to be implemented inside *LightViT* class. 

### B) Positional Embedding

To preserve the context of an image, positional embeddings are associated with each image patch. Positional embeddings encodes the patch positions using sinusoidal waves, however, there are other techniques. 
We follow the definition of positional encoding in the original transformer paper of [Vaswani et. al](https://arxiv.org/abs/1706.03762), which sinusoidal waves. 
To implement a function that creates embeddings for each coordinate of every image patch.

*Output* Inside *LightViT* class, implement a function that fetches the embedding and encapuslate it inside a non-learnable parameter.
"""

def get_pos_embeddings(emb_size, patches_with_clstokens):
    d_model=emb_size
    length=patches_with_clstokens.shape[1]
    b=patches_with_clstokens.shape[0]
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
        
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    pe = pe.unsqueeze(0)

    pe = repeat(pe, '() n e -> b n e', b=b)
    return pe

"""## 3. Encoder Block 

### A) Layer Normalization
[Layer normailzation](https://arxiv.org/abs/1607.06450), similar to other techniques, normalizes an input across the layer dimension by subtracting mean and dividing by standard deviation. One can instantiate layer normalization which has a dimension *d* using [PyTorch built-in function](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).
### B) MHSA
  
 The attention module derives an attention value by measuring similarity between one patch and the other patches. To this end, an image patch with dimension *d* is linearly mapped to three vectors; query **q**, key **k**, and value **v** , hence a distint linear layer should be instantiated to get each of the three vectors. To quantify attention for a single patch, first, the dot product is computed between its **q** and all of the **k** vectors and divide by the square root of the vector dimension i.e. *d* = 8. 
 The result is passed through a softmax layer to get *attention features* and finally multiple with **v** vectors associated with each of the **k** vectors and sum up to get the result. This allows to get an attention vector for each patch by measuring its similarity with other patches.
 
Note that this process should be repeated **N** times on each of the **H** sub-vectors of the 8-dimensional patch, where **N** is the total number of attention blocks. In our case, let **N** = 2, hence, we have 2 sub-vectors, each of length 4. 
The first sub-vector is processed by the first head and the second sub-vector is process by the second head, each head has distinct Q,K, and V mapping functions of size 4x4. 
 
For more information about MHSA, you may refer to this [post](https://data-science-blog.com/blog/2021/04/07/multi-head-attention-mechanism/).
 
"""

class MHSA(nn.Module):
    def __init__(self, d, num_heads): # d: dimension of embedding spacr, n_head: dimension of attention heads
        super(MHSA, self).__init__()
        self.emb_size = d
        self.num_heads = num_heads

        self.keys = nn.Linear(self.emb_size, self.emb_size)
        self.queries = nn.Linear(self.emb_size, self.emb_size)
        self.values = nn.Linear(self.emb_size, self.emb_size)

        self.projection = nn.Linear(self.emb_size, self.emb_size)
        
    def forward(self, x : Tensor) -> Tensor:
        # Sequences has shape (N, seq_length, token_dim)
        # Shape is transformed to   (N, seq_length, n_heads, token_dim / n_heads)
        # And finally we return back    (N, seq_length, item_dim)  (through concatenation)
        
        # split keys, queries and values in num_heads
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values  = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
 
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

"""### C) Residual Connection

Residual connections (also know as skip connections) add the original input to the processed output by a network layer e.g. encoder. They have proven to be useful in deep neural networks as they mitigate problems like exploding / vanishing gradients. 
In transformer, the residual connection is adding the original input to the output from LN &rarr; MHSA. All of the previous operations could be implemented inside a seperate encoder class.

The last part of an encoder, is to a insert another residual connection between the input to the encoder and the output from the encoder passed through another layer of LN &rarr; MLP. 
The MLP consists of 2 layers with hidden size 4 times larger than *d*.

*output*: The output from a single encoder block should have the same dimension as input.
"""

class ViTEncoder(nn.Module):
    def __init__(self, hidden_d, n_heads):
        
        super(ViTEncoder, self).__init__()
        
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MHSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, 4 * hidden_d),
            nn.GELU(),
            nn.Linear(4 * hidden_d, hidden_d)
        )


    def forward(self, x):
        res1 = x
        x = self.mhsa(self.norm1(x))
        x = x + res1  #Residual connection 1
        
        res2 = x
        x = self.mlp(self.norm2(x))
        out = x + res2  #Residual connection 2
        
        return out


"""## 4. Classification Head

The final part of implemeting a transformer is adding a classification head to the model inside *LightViT* class. One can simply use a linear classifier i.e. a linear layer that accepts input of dimension *d* and outputs logits with dimension set to the number of classes for the classification problem at hand.

## 5. Model Train

At this point you have completed the major challenge of the assignment. Now all you need to do is to implement a standard script for training and testing the model. We recommend to use Adam optimizer with 0.005 learning rate and train for 5 epochs.
Example shown below :::
"""

"""
## Define Dataloader (Example shown below)
# mnist_train_dataset = torchvision.datasets.MNIST(root='./data', train = True, transform=torchvision.transforms.ToTensor(), download=True)
# mnist_train_dataloader = torch.utils.data.DataLoader(mnist_train_dataset, shuffle=True, batch_size=64)
# 
# mnist_test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
# mnist_test_dataloader = torch.utils.data.DataLoader(mnist_test_dataset, shuffle=False, batch_size=64)
# 
# for images, labels in mnist_train_dataloader:
#     image_dim = images.shape
#     break

# ## Define Model (Example shown below)
# model = LightViT(image_dim, n_patches=7, n_encoder_blocks=2, d=16, n_heads=8, num_classes=10)
# model = model.to(device)

## Define Optimizer (Example shown below)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

## Define Loss (Example shown below)
# loss_criterion = torch.nn.CrossEntropyLoss()
"""

## Train Function
## Outputs train_accuracies, train_losses, test_accuracies, test_losses
def train_model(model, train_dataloader, train_dataset, test_dataloader, test_dataset, num_epochs, optimizer, loss_criterion):
    model = model.to(device)
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(num_epochs):
        train_loss = 0
        train_correct = 0
        train_total = 0
        for images, labels in train_dataloader:
            
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = loss_criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(train_loss / len(train_dataset))
        train_accuracies.append(train_accuracy)

        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_dataloader:
                images, labels = images, labels
                outputs = model(images)
                loss = loss_criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
            test_accuracy = 100 * test_correct / test_total
            test_accuracies.append(test_accuracy)
            test_losses.append((test_loss) / (len(test_dataset)))
        print("\nEpoch:",epoch," completed. Train accuracy:",train_accuracies[epoch],", Train loss", train_losses[epoch])
        print("Test accuracy:", test_accuracies[epoch], ", Test loss:", test_losses[epoch])
    return train_accuracies, train_losses, test_accuracies, test_losses

# # Train and test (Example shown below)
# num_epochs = 10
# train_accuracies, train_losses, test_accuracies, test_losses = train_model(model, mnist_train_dataloader, mnist_train_dataset, mnist_test_dataloader, mnist_test_dataset, num_epochs, optimizer, loss_criterion)

def plot_accuracy(train_accuracies, test_accuracies, num_epochs):
    plt.plot(np.arange(0,num_epochs), train_accuracies, "r", label="Training Accuracy")
    plt.plot(np.arange(0,num_epochs), test_accuracies, "b", label="Test Accuracy")
    plt.title("Training and Test Accuracies")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.show()

def plot_loss(train_losses, test_losses, num_epochs):
    plt.plot(np.arange(0,num_epochs), train_losses, "r",  label="Training Loss")
    plt.plot(np.arange(0,num_epochs), test_losses, "b",  label="Test Loss")
    plt.title("Training and Test Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss/ Error")
    plt.legend()
    plt.show()

