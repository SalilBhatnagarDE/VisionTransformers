# VisionTransformers
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

In this repository, I release models from the paper:

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

## Use this repository to train Vision Transformers on custom datasets
Using this repository, one can train the Vision Transformer from scratch. 

## Installation and steps to follow
Please follow below steps in order install it and train it on custom dataset :
1. pip install git+https://github.com/SalilBhatnagarDE/VisionTransformers.git 
2. 
