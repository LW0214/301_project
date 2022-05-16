# 301_project

This projects builds upon Yoon Kim’s [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882), showing that w2v embedding with cnn inception network achieves neat performance on text classification while being fast and efficient. Thus I'd like to investigate how other structures of cnn perofrm on text classification, like Lenet and the simplest structure of cnn with just 1 convolution layer, why certain structures work, and how do different embedding layers impact this task as well.

`lenet_normal_cnn.ipynb` contains results of using lenet and the simplest cnn structure on text classification, both with the normal embedding layer as well as pretrained word2vec embedding.

The result shows that cnn sequential depth don't matter that much for text classification task, as the simplest cnn model already achieve even slightly higher performance than using lenet:

lenet:

<img width="210" alt="image" src="https://user-images.githubusercontent.com/66658063/168525317-cb7c720a-80ad-4690-8ec9-8fcda07a542b.png">

simplest cnn:

<img width="217" alt="image" src="https://user-images.githubusercontent.com/66658063/168525343-192171a7-5ff5-4560-864d-6ac8add6d355.png">


`clean_normal_embed.ipynb` contains results of using normal embedding trained from scratch, with inception cnn.

`clean_w2v`: using pre-trained w2v with inception cnn.

The above 2 files show that using the pre-trained w2v achieves slightly better performance than using the embedding layer trained from scratch.

w2v:

<img width="351" alt="image" src="https://user-images.githubusercontent.com/66658063/168525469-94f412b9-f076-4670-9671-e5ad6082006e.png">


normal embedding:

<img width="253" alt="image" src="https://user-images.githubusercontent.com/66658063/168525483-bef48f99-7730-455b-80a0-94b5e16c079d.png">


`dirty_tokenization` folder are experimentations by using a rather dirty tokenization. 

<img width="183" alt="image" src="https://user-images.githubusercontent.com/66658063/168524554-25db9dfc-8990-47ca-b31f-1d582bd4496c.png">

The folder contains the following files:
`normal_cnn_trainable_false.ipynb`: using pretrained w2v and inception cnn

`dirty_embed_trainable_true.ipynb`: using embedding initiated from scratch and inception cnn

The results show that a dirty tokenization method results in lower accuracy (0.78 for w2v) and takes much longer time to train (30 epochs) to converge.

<img width="276" alt="image" src="https://user-images.githubusercontent.com/66658063/168524784-b2348765-ce70-4da1-9e60-096eaf7d52be.png">

`tune_normal_embed.ipynb` contains a failed attempt of trying to conduct hyperband on cnn inception for text classification. The result is likely due to the large size of the embedding layer. Even setting the embedding dimension to just 100 still results in a error on resource draining in hyperband.

Hyperband appears to work with setting embedding size to just 32, but the tradeoff is having really poor accuracy, too the point of being meaningless.

<img width="172" alt="image" src="https://user-images.githubusercontent.com/66658063/168525185-139025d7-558a-440f-acc5-8841c8351ce9.png">

<img width="148" alt="image" src="https://user-images.githubusercontent.com/66658063/168525189-847e2d5c-54de-4a79-8d7a-0740532f36ae.png">


`other_tests` is a folder that contains some other failure tests.
Inside it is a folder named  `overfitting`. In it are 2 notebooks of simplest cnn structure and inception cnn structure that results in high performance (accuracy of 88) but extreme overfitting 

<img width="310" alt="image" src="https://user-images.githubusercontent.com/66658063/168524869-9e08659d-b55f-47db-9fc0-e7294bb9d6b3.png">

The reason is likely due to not adding the "L2 regularization" term, ephasizing the importance of regularization in this task.
