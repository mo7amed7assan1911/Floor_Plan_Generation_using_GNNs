# Gaduation Project: Residential Floor Plan Generation Using Deep Learning Techniques

> Our project provides a user-friendly software solution that minimizes the gap between the complexity of designing residential floor plans and the capabilities of non-technical users. users can input the boundary of their piece of ground and their preferences which are then seamlessly processed by our advanced AI model to generate customized floor plans. 


![image](https://github.com/mo7amed7assan1911/Floor_Plan_Generation_using_GNNs/assets/55090589/6fa02b98-ebc2-4282-b3f9-fb056de70171)

## Dataset
- The source dataset is [Rplan Dataset](http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html), it is a dataset consists of about 80k floor plans as images.
- First, we converted the image-based RPlan dataset to a geometry-based dataset, this conversion helped us with some benefits:
  - Accurate Representation.
  - Geometric Operations such as buffers, and intersections.
- Then, we created a customized dataset as **Graphs** to be a new version of the Rplan floor plans. This conversion helped us train the **GAT-Net** model. you can see this work from this [notebook](https://github.com/mo7amed7assan1911/Floor_Plan_Generation_using_GNNs/blob/with-boundary/Creating_Dataset/generating-graphs.ipynb)

## Our full architecture consists of 2 stages:
1. Predicting room centroids. Using CNN model.
2. Best room size estimation Using GNN model **[Our work in this repo]**

> Below, is the full architecture we used to take user input to generate his final layout.

![image](https://github.com/mo7amed7assan1911/Floor_Plan_Generation_using_GNNs/assets/55090589/c0448e53-fdc6-471a-9e0c-8b1aee185363)

* User inputs:
  * The boundary of his piece of ground and the position of the front door.
  * His preferences such as the number of rooms, bathrooms, and kitchens.
* User gets:
   * Final layout for his floor plan.
   * 3D model for his floor plan.

> **We will focus in this repository on the second stage to get the best room size estimations using the GNN model.**
---
## Preprocessing

#### Creation of our customized dataset [Graph Based RPlan]:
* As we said before, the Rplan dataset is an image-based dataset and we converted it to Geometric based using the idea of contours and convert each one to a polygon. Now we will convert these polygons into graphs.
* For each floor plan two types of graphs were created: one representing the floor plan itself and another representing the boundary of the floor plan.
* The process of creating the graphs involved the following steps:
	* **Polygon-to-Node Conversion:** The nodes were assigned features such as the type of polygon (e.g., room, bathroom), the centroid's X and Y coordinates, the room size (expressed as the ratio of the room's area to the total floor plan area), and the width and height of the square bounding the polygon. Also, the boundary graph was created.
	* **Connections - Edges - between the nodes:** Real connections, Living to all connections, and All-to-All connections.

> You could see how this work is done from this [notebook](https://github.com/mo7amed7assan1911/Floor_Plan_Generation_using_GNNs/blob/with-boundary/Creating_Dataset/generating-graphs.ipynb).
> 
> Below, is a floor plan from the source dataset as images and the corresponding generated graphs.

![image](https://github.com/mo7amed7assan1911/Floor_Plan_Generation_using_GNNs/assets/55090589/3e49c78e-f1e5-49a6-8dd3-23b1ce8151f1)
> **We utilized the Living-to-All connection type in our model** since the coming centroids from the CNN model lack information about the real connections between these nodes.
>
> To compensate for the model's reduced ability to learn from real neighboring rooms, we incorporated **attention mechanisms** in our layers. These mechanisms aid in identifying the crucial nodes for each node, enabling the model to focus on relevant information.

#### A pre-model to get intuitions before designing our GAT-Net model to get room size estimations.
> * During our experimentation, for classifying room types model we compared the performance of two types of graph convolutional layers: Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT) in 50 epochs
>     * GCN achieved 68% accuracy.
>     * GAT achieved 94% accuracy, due to its attention mechanism.
>  
> For this reason and the usage of the attention mechanism, I decided to use the GATConv layers as the core of our model.
> 
>  You can see how this work is done from this [notebook](https://github.com/mo7amed7assan1911/Floor_Plan_Generation_using_GNNs/blob/with-boundary/Classifying_node_types/classifying_node_types.ipynb)

#### How we handled the problem of Over Smoothing after just 2 layers in Graph Neural Networks
> To address this, we introduced the concept of residual connections. However, instead of employing skip connections, we utilized CONCATENATION connections to preserve and retain the essential features during the learning process.
---
## GAT-Net model: A Residual Graph Attention Network for Floor Plan Analysis.
> Name GAT-Net because The core of our model is the Graph Attention Network (GAT) architecture, customized for floor plan analysis, and to highlight its ability to extract meaningful insights from floor plan graphs using attention mechanisms.

![image](https://github.com/mo7amed7assan1911/Floor_Plan_Generation_using_GNNs/assets/55090589/8df287fe-252f-4884-aa90-a21eeb1bb118)

### Train process
* The training process of GAT-Net involved training the model on an Nvidia P100 GPU with specific hyperparameters:
	* Number of Epochs = 300 epoc epo.
	* Learning Rate = 0.001
	* Optimizer: Adam
	* weight decay: 3e-5
	* Loss: Mean Square Error (MSE)
	* LR Scheduler gamma value: 0.950.

> You could see how the GAT-Net is designed and implemented from this [notebook](https://github.com/mo7amed7assan1911/Floor_Plan_Generation_using_GNNs/blob/with-boundary/GAT-Net_model/GAT-Net_Model.ipynb).
>
> Or you could run directly the `main.py` file which needs inputs from the CNN model, but I provided an example inside it.
---
### Results of the GAT-Net model on the test set
![image](https://github.com/mo7amed7assan1911/Floor_Plan_Generation_using_GNNs/assets/55090589/b05d85ff-c45a-4f2e-9ac3-163819c2ebdc)
> You could see how we tested our model on the test set from this [notebook](https://github.com/mo7amed7assan1911/Floor_Plan_Generation_using_GNNs/blob/with-boundary/GAT-Net_model/Testing_GAT-Net_model.ipynb)
---

## Getting Started
To get started with the project, you need to install the required dependencies.
> Thses steps will help you run the `main.py` and User_constrains_to_floor_plan.ipynb files. As, they don't need the dataset to be downloaded.

### Dependencies
* Python > 3.9
* requirements.txt

### Installation
1. Clone the repo
```bash
git clone https://github.com/mo7amed7assan1911/Floor_Plan_Generation_using_GNNs.git
cd Floor_Plan_Generation_using_GNNs
```
2. install the required dependencies
```bash
pip install -r requirements.txt
```
### For Docker users
Run the following commands to build the docker image and run the container
```bash
docker run mo7amed7assan1911/floor-plan-generation-using-gnns:latest
```
## How to test the GAT-Net model

> * You could test the model on the testset from this [notebook](GA-Net_model/Testing_GAT-Net_model.ipynb).
>
> * You could use the [User_constrains_to_floor_plan.ipynb](User_constrains_to_floor_plan.ipynb) notebook to get the user inputs and then test the model on them.
>
> * You could directly run the `main.py` file which needs inputs from the CNN model, but I provided an example inside it.

