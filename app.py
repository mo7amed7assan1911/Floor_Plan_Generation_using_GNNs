import streamlit as st
from main import Run
from utils import boundary_to_image, get_user_inputs_as_image, draw_both_graphs
import os
from PIL import Image

def get_example(example_name):
    if example_name == 'EX 1':
        boundary_wkt = "POLYGON ((25.599999999999994 65.32413793103447, 200.38620689655173 65.32413793103447, 200.38620689655173 75.91724137931033, 230.4 75.91724137931033, 230.4 190.67586206896553, 67.97241379310344 190.67586206896553, 67.97241379310344 176.55172413793102, 25.599999999999994 176.55172413793102, 25.599999999999994 65.32413793103447))"
    
        front_door_wkt = "POLYGON ((38.436315932155225 179.69850789734912, 63.610586007499926 179.69850789734912, 63.610586007499926 176.55172413793102, 38.436315932155225 176.55172413793102, 38.436315932155225 179.69850789734912))"
        
        # Data of the inner rooms or bathrooms
        room_centroids  = [(201, 163), (193, 106)]
        bathroom_centroids = [(91, 91), (52, 95)]
        kitchen_centroids = [(137, 89)]
    
    elif example_name == 'EX 2':
        boundary_wkt = "POLYGON ((29.043431083399053 47.85822973601723, 179.28640942886287 47.85822973601723, 179.28640942886287 75.73837004754658, 230.4 75.73837004754658, 230.4 208.94348486929798, 58.47246807890228 208.94348486929798, 58.47246807890228 162.4765843500824, 29.043431083399053 162.4765843500824, 29.043431083399053 47.85822973601723))"
        
        front_door_wkt = "POLYGON ((32.48686216679811 47.05651513070204, 25.599999999999994 47.05651513070204, 25.599999999999994 74.60396379789447, 29.043431083399053 74.60396379789447, 29.043431083399053 47.85822973601723, 32.48686216679811 47.85822973601723, 32.48686216679811 47.05651513070204))"
        
        # Data of the inner rooms or bathrooms
        room_centroids  = [(203, 101), (191, 171)]
        bathroom_centroids = [(83, 182), (152, 66)]
        kitchen_centroids = [(52, 131)]
        
    elif example_name == 'EX 3':
        
        boundary_wkt = "POLYGON ((44.92075471698112 53.615094339622644, 230.4 53.615094339622644, 230.4 78.73207547169811, 209.14716981132074 78.73207547169811, 209.14716981132074 202.38490566037737, 25.599999999999994 202.38490566037737, 25.599999999999994 71.00377358490566, 44.92075471698112 71.00377358490566, 44.92075471698112 53.615094339622644))"
        
        front_door_wkt = "POLYGON ((212.46404317939437 194.3995689439927, 212.46404317939437 167.86458199940353, 209.14716981132074 167.86458199940353, 209.14716981132074 194.3995689439927, 212.46404317939437 194.3995689439927))"
        
        # Data of the inner rooms or bathrooms
        room_centroids  = [(184, 77), (67, 94)]
        bathroom_centroids = [(185, 134), (126, 74)]
        kitchen_centroids = [(52, 176)]

    elif example_name == 'EX 4':
        boundary_wkt = "POLYGON ((58.18181818181817 69.85672370603027, 230.4 69.85672370603027, 230.4 105.54157219087877, 211.7818181818182 105.54157219087877, 211.7818181818182 200.183996433303, 25.599999999999994 200.183996433303, 25.599999999999994 58.99611764542421, 58.18181818181817 58.99611764542421, 58.18181818181817 69.85672370603027))"
        
        front_door_wkt = "POLYGON ((56.16288055733307 55.816003566697006, 30.721967927515397 55.816003566697006, 30.721967927515397 58.99611764542421, 56.16288055733307 58.99611764542421, 56.16288055733307 55.816003566697006))"
        
        # Data of the inner rooms or bathrooms
        room_centroids  = [(198, 87), (174, 166)]
        bathroom_centroids = [(51, 169), (148, 91)]
        kitchen_centroids = [(44, 105)]
        
        
    return boundary_wkt, front_door_wkt, room_centroids, bathroom_centroids, kitchen_centroids

def get_input_images(boundary_wkt, front_door_wkt, room_centroids, bathroom_centroids, kitchen_centroids):
    boundary_img_path = boundary_to_image(boundary_wkt, front_door_wkt)
    boundary_img = Image.open(boundary_img_path)
    
    user_inputs_img_path = get_user_inputs_as_image(boundary_wkt, front_door_wkt, room_centroids, bathroom_centroids, kitchen_centroids)
    user_inputs_img = Image.open(user_inputs_img_path)
    
    return boundary_img, user_inputs_img

def main():
    
    st.sidebar.write("""
            # Floor Plan 🏠 Generator Using Graph Neural Networks (GNNs)
            [![GitHub](https://img.shields.io/badge/GitHub-mo7amed7assan1911-blue?style=flat-square&logo=github)](https://github.com/mo7amed7assan1911/Floor_Plan_Generation_using_GNNs/blob/with-boundary/README.md)
            
            [![LinkedIn](https://img.shields.io/badge/LinkedIn-mo7amed1911-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/mo7amed1911)
            
            Welcome 🙋🏻‍♂️ to the second part of our project, **Designing Residential Floor Plans**, where we explore the realm of Graph Neural Networks (GNNs) to craft an advanced Floor Plan Generator. In this stage, we harness the strength of GNNs to process the centroids from the initial model, enabling us to generate the final layout while accurately estimating the optimal width and height of rooms, bathrooms, and kitchens.       
            > I provided 4 examples of user inputs with first model's outputs on it, you can select from them and see how our model works.
            >
            > **User could select an example from the sidebar. 👈**
            
             > **Note:** The user can edit the centroids of the rooms, bathrooms, and kitchens if he needs to test the model.
            
            """)
    
    with st.expander("A Deep Dive into Our Graph-Based Floor Plan Generator"):
        st.write("""
            ## Gaduation Project: Residential Floor Plan Generation Using Deep Learning Techniques
            > Our project provides a user-friendly software solution that minimizes the gap between the complexity of designing residential floor plans and the capabilities of non-technical users. users can input the boundary of their piece of ground and their preferences which are then seamlessly processed by our advanced AI model to generate customized floor plans. 
            """)
        
        st.image("https://github.com/mo7amed7assan1911/Floor_Plan_Generation_using_GNNs/assets/55090589/6fa02b98-ebc2-4282-b3f9-fb056de70171", use_column_width=True)
        
        st.write("""
        ### Dataset
        - The source dataset is [Rplan Dataset](http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html), it is a dataset consists of about 80k floor plans as images.
        - First, we converted the image-based RPlan dataset to a geometry-based dataset, this conversion helped us with some benefits:
            - Accurate Representation.
            - Geometric Operations such as buffers, and intersections.
        - Then, we created a customized dataset as **Graphs** to be a new version of the Rplan floor plans. This conversion helped us train the **GAT-Net** model. you can see this work from this [notebook](https://github.com/mo7amed7assan1911/Floor_Plan_Generation_using_GNNs/blob/with-boundary/Creating_Dataset/generating-graphs.ipynb)

        ### Our full architecture consists of 2 stages:
        1. Predicting room centroids. Using CNN model.
        2. Best room size estimation Using GNN model **[Our work in this repo]**

        > Below, is the full architecture we used to take user input to generate his final layout.
        """)
        
        st.image("https://github.com/mo7amed7assan1911/Floor_Plan_Generation_using_GNNs/assets/55090589/c0448e53-fdc6-471a-9e0c-8b1aee185363", use_column_width=True)
        st.write("""

        * User inputs:
            * The boundary of his piece of ground and the position of the front door.
         * His preferences such as the number of rooms, bathrooms, and kitchens.
        * User gets:
            * Final layout for his floor plan.
            * 3D model for his floor plan.

        > **We will focus in this repository on the second stage to get the best room size estimations using the GNN model.**
        ---
        ### Preprocessing

        #### Creation of our customized dataset [Graph Based RPlan]:
        * As we said before, the Rplan dataset is an image-based dataset and we converted it to Geometric based using the idea of contours and convert each one to a polygon. Now we will convert these polygons into graphs.
        * For each floor plan two types of graphs were created: one representing the floor plan itself and another representing the boundary of the floor plan.
        * The process of creating the graphs involved the following steps:
            * **Polygon-to-Node Conversion:** The nodes were assigned features such as the type of polygon (e.g., room, bathroom), the centroid's X and Y coordinates, the room size (expressed as the ratio of the room's area to the total floor plan area), and the width and height of the square bounding the polygon. Also, the boundary graph was created.
            * **Connections - Edges - between the nodes:** Real connections, Living to all connections, and All-to-All connections.

        > You could see how this work is done from this [notebook](https://github.com/mo7amed7assan1911/Floor_Plan_Generation_using_GNNs/blob/with-boundary/Creating_Dataset/generating-graphs.ipynb).
        > 
        > Below, is a floor plan from the source dataset as images and the corresponding generated graphs.
        """)
        
        st.image("https://github.com/mo7amed7assan1911/Floor_Plan_Generation_using_GNNs/assets/55090589/3e49c78e-f1e5-49a6-8dd3-23b1ce8151f1")

        st.write("""
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
        """)
        
        st.image("https://github.com/mo7amed7assan1911/Floor_Plan_Generation_using_GNNs/assets/55090589/8df287fe-252f-4884-aa90-a21eeb1bb118")
        
        st.write("""


        ### Train process
        * The training process of GAT-Net involved training the model on an Nvidia P100 GPU with specific hyperparameters:
            * Number of Epochs = 300 epoc epo.
            * Learning Rate = 0.001
            * Optimizer: Adam
            * weight decay: 3e-5
            * Loss: Mean Square Error (MSE)
            * LR Scheduler gamma value: 0.950.
        > Below, is the training chart
        """)
        
        st.image("https://github.com/mo7amed7assan1911/Floor_Plan_Generation_using_GNNs/assets/55090589/e9dbaf2d-b3d2-4cbd-8883-691099102aad")

        st.write("""
        > You could see how the GAT-Net is designed and implemented from this [notebook](https://github.com/mo7amed7assan1911/Floor_Plan_Generation_using_GNNs/blob/with-boundary/GAT-Net_model/GAT-Net_Model.ipynb).
        >
        > Or you could run directly the `main.py` file which needs inputs from the CNN model, but I provided an example inside it.
        ---
        ### Results of the GAT-Net model on the test set
        """)
        
        st.image("https://github.com/mo7amed7assan1911/Floor_Plan_Generation_using_GNNs/assets/55090589/b05d85ff-c45a-4f2e-9ac3-163819c2ebdc")
        
        st.write("""
        
        > You could see how we tested our model on the test set from this [notebook](https://github.com/mo7amed7assan1911/Floor_Plan_Generation_using_GNNs/blob/with-boundary/GAT-Net_model/Testing_GAT-Net_model.ipynb)
        ---

        #### You could see results from the whole architecture based on the user inputs from the `Outputs folder` [here](https://github.com/mo7amed7assan1911/Floor_Plan_Generation_using_GNNs/tree/with-boundary/Outputs)
        #### Or you could directly run the `main.py` file that takes inputs from the CNN model. But don't worry I provided an example in it.

                 
                 """)
        
    st.write('---')
    
    # User selects and example
    example_name = st.sidebar.selectbox("Select an example", ['EX 1', 'EX 2', 'EX 3', 'EX 4'])
    
    # Get the data of the example user selected
    boundary_wkt, front_door_wkt, room_centroids, bathroom_centroids, kitchen_centroids = get_example(example_name)
    
    # Convert the boundary and front door to image
    boundary_img, user_inputs_img = get_input_images(boundary_wkt, front_door_wkt, room_centroids, bathroom_centroids, kitchen_centroids)
    
    left_column, right_column = st.columns(2)
    
    with left_column:
        st.write(
                    "<div style='text-align: center;'>User Inputs</div>",
                    unsafe_allow_html=True
                )
        st.write('---')
        st.image(boundary_img,use_column_width=True)
    
    with right_column:
        # st.write('**The Output of first model**', width=500)
        st.write(
                    "<div style='text-align: center;'>Outputs of the first model</div>",
                    unsafe_allow_html=True
                )
        st.write('---')
        room_input = st.text_input("Room centroids", value=room_centroids)
        bath_input = st.text_input("Bathroom centroids", value=bathroom_centroids)
        kitchen_input = st.text_input("Kitchen centroids", value=kitchen_centroids)
    
    # Get model output
    output_path, Graph_netorkX, Boundary_networkX = Run(boundary_wkt, front_door_wkt, room_centroids, bathroom_centroids, kitchen_centroids)
    graphs_image_path = draw_both_graphs(Graph_netorkX, Boundary_networkX)
    graph_image = Image.open(graphs_image_path)
    output_image = Image.open(output_path)
    
    with st.expander("See how we pre-process the user inputs"):
        # Create two columns inside the expander
        col1, col2 = st.columns(2)

        # Add content to the first column
        with col1:
            st.write("Concatinate centroids with the boundary")
            st.image(user_inputs_img,use_column_width=True)
            
        # Add content to the second column
        with col2:
            st.write("Converting these info to **Graph**")
            st.image(graph_image,use_column_width=True)
    
    st.write("""
             ## Output of the GAT-Net model
             """)
    st.image(output_image,use_column_width=True)
    # left_column, middle_column, right_column = st.columns(3)
    # with left_column:
    #     st.write('## Room Centroids')
    #     room_input = st.text_input("You can edit the centroids of the rooms if you need", value=room_centroids)
    # with middle_column:
    #     st.write('## Bathroom Centroids')
    #     bath_input = st.text_input("You can edit the centroids of the bathrooms if you need", value=bathroom_centroids)
    # with right_column:
    #     st.write('## Kitchen Centroids')
    #     kitchen_input = st.text_input("You can edit the centroids of the kitchens if you need", value=kitchen_centroids)
        
    # if st.button("Submit"):
    #     image_name = Run(user_inputs)
    #     image_path = 'Outputs/' + image_name
        
    #     st.image(image_path, caption='Generated Graph', use_column_width=True)

if __name__ == "__main__":
    main()
