# Vision_Transformer
by Alexey Dosovitskiy\*†, Lucas Beyer\*, Alexander Kolesnikov\*, Dirk
Weissenborn\*, Xiaohua Zhai\*, Thomas Unterthiner, Mostafa Dehghani, Matthias
Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit and Neil Houlsby\*†.
(\*) equal technical contribution, (†) equal advising.

Vision Transformers are the state of the art methods for classification or object detection problems.

![Figure 1 from paper](vit_figure.png)

Overview of the model: we split an image into fixed-size patches, linearly embed
each of them, add position embeddings, and feed the resulting sequence of
vectors to a standard Transformer encoder. In order to perform classification,
we use the standard approach of adding an extra learnable "classification token"
to the sequence.

## In this repository, I release models from the paper:

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

## Use this repository to train Vision Transformers on custom datasets
Using this repository, one can train the Vision Transformer from scratch. 

There are no pretrained weigths as of now in this repository. You can train on custom datasets from scratch.

## Installation and steps to follow
Please follow the below steps in order to install Vision_Transformer and train it on custom datasets :
#### 1. pip install git+https://github.com/SalilBhatnagarDE/VisionTransformers.git 

#### 2. import Vision_Transformer

#### 3. Define your datasets and dataloaders. 
- An example is shown below :
- import torchvision
- import torch
- mnist_train_dataset = torchvision.datasets.MNIST(root='./data', train = True, transform=torchvision.transforms.ToTensor(), download=True)
- mnist_train_dataloader = torch.utils.data.DataLoader(mnist_train_dataset, shuffle=True, batch_size=64)
- mnist_test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
- mnist_test_dataloader = torch.utils.data.DataLoader(mnist_test_dataset, shuffle=False, batch_size=64)

#### 4. Select device for training => device = torch.device("cpu")

#### 5. Make a model object using class Vision_Transformer.LightViT :
- model = Vision_Transformer.LightViT(image_dim=(1,28,28), n_patches=7, n_encoder_blocks=1, d=8, n_heads=4, num_classes=10)
- image_dim is the input dimension of the images
- n_patches are the number of patches you want to have in a single input image
- n_encoder_blocks are the number of encoders blocks you want to stack one over the other having self attension MSHA modules in each of them
- d is the dimension of the input vectors you want to have in encoders. For each patch, your pixels in that patch will be linearly mapped to d dimension
- n_heads are the number of attention heads you want to have
- num_classes depends upon your classification problem
- All the above arguments need to pass through Vision_Transformer.LightViT to initiate the object model.

#### 6. Define optimizer, loss_criterion and number of epochs and pass the arguments to function train_model

#### 7. Example : 
- train_accuracies, train_losses, test_accuracies, test_losses = Vision_Transformer.train_model(model, train_dataloader=mnist_train_dataloader, train_dataset=mnist_train_dataset, test_dataloader=mnist_test_dataloader, test_dataset=mnist_test_dataset, num_epochs=5, optimizer=torch.optim.Adam(model.parameters(), lr=0.005), loss_criterion=torch.nn.CrossEntropyLoss())

#### 8. To plot training curves, use below code :
- Vision_Transformer.plot_accuracy(train_accuracies, test_accuracies, num_epochs=5)
- Vision_Transformer.plot_loss(train_losses, test_losses, num_epochs=5)
