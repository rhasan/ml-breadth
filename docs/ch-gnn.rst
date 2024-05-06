.. _gnn_basics:

.. contents::
    :local:
    :depth: 2

Introduction to Graph Neural Networks with PyTorch
===================================================

Graph Neural Networks (GNNs) have emerged as a powerful tool for machine learning tasks on graph-structured data. In this tutorial, we will introduce the concept of GNNs and demonstrate how to implement a simple GNN using PyTorch.

First, let's install the necessary libraries:

..  code-block:: bash

    pip install torch torch-geometric

This will install PyTorch and PyTorch Geometric. PyTorch Geometric is a library for geometric deep learning (graph neural networks) built on top of PyTorch. We will only use a dataset from this library in this tutorial. We will code our GNN from scratch.

What are Graph Neural Networks?
------------------------------------
Traditional machine learning algorithms assume that instances are `independently and identically distributed (i.i.d.) <https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables>`_. However, in many real-world applications, data instances are not isolated but exist in the form of graphs, where relationships between instances matter.
Graph Neural Networks are a type of neural network designed to perform inference on data structured as graphs. They extend traditional neural networks to handle graph data, capturing the dependencies between connected nodes in a graph.

Implementing a Simple GNN with PyTorch
------------------------------------
We will code a node classification GNN. To keep things simple, our GNN will consist of just two fully connected layers and will use the adjacency matrix of the graph for message passing. Here's the code for our GNN:

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class GNN(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(GNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x, adjacency):
            x = torch.mm(adjacency, x)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            return x

Loading the Dataset and Preprocessing
------------------------------------
We will use the KarateClub dataset from the PyTorch Geometric library to train our GNN. This is a small graph dataset representing a social network of 34 members of a karate club and their relationships.
The Karate Club dataset is a network of 34 members of a karate club, documenting links between pairs of members who interacted outside the club. It has become a standard benchmark for testing and understanding graph theory algorithms because of its relatively small size and the interesting social dynamics it captured.

Each node in the dataset represents a member of the club. The nodes have features associated with them which are 34-dimensional binary vectors indicating whether a member has interacted with another member outside the club. Each feature corresponds to a member and the feature value is 1 if the interaction occurred, otherwise it's 0. So, the node features encode the social interaction network of the club.

The labels associated with each node are based on the communities formed after a disagreement led to a split in the club. The members formed four distinct communities, each labeled as 0, 1, 2, or 3 in the dataset. These labels represent the 'Administrator Community', 'Instructor Community', 'Member Community', and 'Newcomer Community', respectively. The 'Administrator Community' (0) and 'Instructor Community' (1) are the ones formed around the original administrator and the instructor. The 'Member Community' (2) consists of long-standing members who chose not to align with either the administrator or the instructor, while the 'Newcomer Community' (3) comprises of new members who joined the club around the time of the split. This dataset provides valuable insights into the dynamics of network communities and has become a benchmark for testing graph theory algorithms.

Now let's look at the code for loading and pre-processing the dataset:

.. code-block:: python

    from torch_geometric.datasets import KarateClub
    from torch_geometric.utils import to_dense_adj

    # Load the KarateClub dataset
    dataset = KarateClub()

    # Get the features and labels
    features = dataset[0].x
    labels = dataset[0].y

    # Get the adjacency matrix
    adjacency = to_dense_adj(dataset[0].edge_index)[0]

Training the GNN
----------------
To train our GNN, we define a training loop where we compute the cross-entropy loss between the GNN's predictions and the actual labels, and update the model's parameters using backpropagation and Adam optimization.

.. code-block:: python

    import torch.optim as optim

    # Initialize the model, optimizer and loss function
    model = GNN(input_dim=features.shape[1], hidden_dim=32, output_dim=dataset.num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Number of training epochs
    epochs = 100

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(features, adjacency)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

Making Predictions with the GNN
--------------------------------
Once the model is trained, we can use it to make predictions on unseen data. In our case, we use the trained model to predict the class of each node in the Karate Club graph. By comparing the predicted classes with the actual classes, we can evaluate the performance of our GNN.

.. code-block:: python

    # Set model to evaluation mode
    model.eval()

    # Make predictions
    with torch.no_grad():
        outputs = model(features, adjacency)
        _, predicted = torch.max(outputs, 1)

    # Print the predicted classes
    print('Predicted classes:', predicted.numpy())

    # Print the actual classes
    print('Actual classes:', labels.numpy())

The torch.max function returns the maximum value along a given dimension in a tensor. In this case, we use it to get the index of the maximum value in each row of the outputs tensor, which gives us the predicted class for each node.

Remember to set your model to evaluation mode before making predictions. This disables certain layers and operations like dropout and batch normalization that behave differently during training and evaluation.

Conclusion
--------------------------------
This tutorial introduced the concept of Graph Neural Networks and demonstrated how to implement a simple GNN with PyTorch. While our GNN is basic and doesn't include advanced features like graph convolutions or self-loops, it serves as a good starting point for understanding how GNNs work. For more advanced graph neural network models, consider exploring libraries like PyTorch Geometric or DGL.