from utils import *
from model import *


def get_info():
    """
        Function to return:
            - Boundary: "POLYGON ((105.44810944559121 78 ....
            - front_door: "POLYGON  (105.44810944559121 78 ....
            - room_centroids: [(81, 105), (55, 151), (134, 105)]
            - bathroom_centroids: [(81, 105), (55, 151), (134, 105)]
            - kitchen_centroids: [(81, 105), (55, 151), (134, 105)]
    """
    boundary_wkt = "POLYGON ((58.18181818181817 69.85672370603027, 230.4 69.85672370603027, 230.4 105.54157219087877, 211.7818181818182 105.54157219087877, 211.7818181818182 200.183996433303, 25.599999999999994 200.183996433303, 25.599999999999994 58.99611764542421, 58.18181818181817 58.99611764542421, 58.18181818181817 69.85672370603027))"

    front_door_wkt = "POLYGON ((56.16288055733307 55.816003566697006, 30.721967927515397 55.816003566697006, 30.721967927515397 58.99611764542421, 56.16288055733307 58.99611764542421, 56.16288055733307 55.816003566697006))"

    # Data of the inner rooms or bathrooms
    room_centroids  = [(198, 87), (174, 166)]
    bathroom_centroids = [(51, 169), (155, 91)]
    kitchen_centroids = [(44, 105)]
    
    return boundary_wkt, front_door_wkt, room_centroids, bathroom_centroids, kitchen_centroids


def preProcessing_toGraphs(Boundary, front_door, room_centroids, bathroom_centroids, kitchen_centroids):
    
    Boundary = shapely.wkt.loads(Boundary)
    front_door = shapely.wkt.loads(front_door)
    
    
    # Flip the y axis of all polygons and points
    Boundary = scale(Boundary)
    front_door = scale(front_door)
    room_centroids = [scale(x) for x in room_centroids]
    bathroom_centroids = [scale(x) for x in bathroom_centroids]
    kitchen_centroids = [scale(x) for x in kitchen_centroids]
        
    # Retruning the centroids of the rooms and bathrooms from Point to tuple
    room_centroids = [x.coords[0] for x in room_centroids]
    bathroom_centroids = [x.coords[0] for x in bathroom_centroids]
    kitchen_centroids = [x.coords[0] for x in kitchen_centroids]
            
    living_centroid    = [(Boundary.centroid.x, Boundary.centroid.y)]
    
    user_constraints = {
        'living': living_centroid,
        'room': room_centroids,
        'bathroom': bathroom_centroids,
        'kitchen': kitchen_centroids
    }
    
    # Making networkX graphs [_n for networkX]
    B_n = Handling_dubplicated_nodes(Boundary, front_door)
    G_n = centroids_to_graph(user_constraints, living_to_all=True)
    
    # Converting the networkX graph to pytorch geometric data
    B = from_networkx(B_n, group_node_attrs=['type', 'centroid'], group_edge_attrs=['distance'])
    
    features = ['roomType_embd', 'actualCentroid_x', 'actualCentroid_y']
    G = from_networkx(G_n, group_edge_attrs=['distance'], group_node_attrs=features)
    
    # To use them later for visualization
    
    # Normalization
    G_x_mean = G.x[:, 1].mean().item()
    G_y_mean = G.x[:, 2].mean().item()

    G_x_std = G.x[:, 1].std().item()
    G_y_std = G.x[:, 2].std().item()

    G.x[:, 1:] = (G.x[:, 1:] - torch.tensor([G_x_mean, G_y_mean])) / torch.tensor([G_x_std, G_y_std])
    
    first_column_encodings = F.one_hot(G.x[:, 0].long(), 7)
    G.x = torch.cat([first_column_encodings, G.x[:, 1:]], axis=1)
    
    #### Normalization for the boundary graph
    B_x_mean = B.x[:, 1].mean().item()
    B_y_mean = B.x[:, 2].mean().item()

    B_x_std = B.x[:, 1].std().item()
    B_y_std = B.x[:, 2].std().item()
    
    B.x[:, 1:] = (B.x[:, 1:] - torch.tensor([B_x_mean, B_y_mean])) / torch.tensor([B_x_std, B_y_std])
    
    
    # Befor passing the data to the model
    G.x = G.x.to(torch.float32)
    G.edge_attr = G.edge_attr.to(torch.float32)
    G.edge_index = G.edge_index.to(torch.int64)

    B.x = B.x.to(G.x.dtype)
    B.edge_index = B.edge_index.to(G.edge_index.dtype)
    B.edge_attr = B.edge_attr.to(G.edge_attr.dtype)
    
    
    # Returning back to the original data [not normalized]
    B_not_normalized = B.clone()
    G_not_normalized = G.clone()
    
    G_not_normalized.x[:, -2] = G_not_normalized.x[:, -2] * G_x_std + G_x_mean
    G_not_normalized.x[:, -1] = G_not_normalized.x[:, -1] * G_y_std + G_y_mean

    B_not_normalized.x[:, -2] = B_not_normalized.x[:, -2] * B_x_std + B_x_mean
    B_not_normalized.x[:, -1] = B_not_normalized.x[:, -1] * B_y_std + B_y_mean

    return G, B, G_not_normalized, B_not_normalized, Boundary, front_door






if __name__ == '__main__':
    
    # Get the data
    # Boundary, front_door, room_centroids, bathroom_centroids, kithchen_centroids = get_info()
    
    Boundary = "POLYGON ((58.18181818181817 69.85672370603027, 230.4 69.85672370603027, 230.4 105.54157219087877, 211.7818181818182 105.54157219087877, 211.7818181818182 200.183996433303, 25.599999999999994 200.183996433303, 25.599999999999994 58.99611764542421, 58.18181818181817 58.99611764542421, 58.18181818181817 69.85672370603027))"

    front_door = "POLYGON ((56.16288055733307 55.816003566697006, 30.721967927515397 55.816003566697006, 30.721967927515397 58.99611764542421, 56.16288055733307 58.99611764542421, 56.16288055733307 55.816003566697006))"

    # Data of the inner rooms or bathrooms
    room_centroids  = [(198, 87), (174, 166)]
    bathroom_centroids = [(51, 169), (155, 91)]
    kitchen_centroids = [(44, 105)]
    
    # ========================================================================
    # Preprocessing
    G, B, G_not_normalized, B_not_normalized, Boundary_as_polygon, front_door_as_polygon = preProcessing_toGraphs(Boundary, front_door, room_centroids, bathroom_centroids, kitchen_centroids)
    the_door = Point(B_not_normalized.x[-1][1:].detach().cpu().numpy()).buffer(3)
    #=========================================================================
    # Model
    # model_path = r"D:\Grad\Best models\v2\Best_model_V2.pt"
    model_path = r"D:\Grad\Best models\v3_UnScalled\Best_model_V3.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_model(model_path, device)
    
    #=========================================================================
    # Inference
    prediction    = model(G.to(device), B.to(device))
    w_predicted   = prediction[0].detach().cpu().numpy()
    h_predicted   = prediction[1].detach().cpu().numpy()
    prediction    = np.concatenate([w_predicted.reshape(-1, 1), h_predicted.reshape(-1, 1)], axis=1)
    
    #=========================================================================
    # Rescaling back to the original values
    # G.x[:, -2] = G.x[:, -2] * G_x_std + G_x_mean
    #=========================================================================
    # Visualization
    output = FloorPlan_multipolygon(G_not_normalized, prediction)
    polygons = output.get_multipoly(Boundary_as_polygon, the_door)
    polygons.plot(cmap='twilight', figsize=(4, 4), alpha=0.8, linewidth=0.8, edgecolor='black');
    plt.savefig('hello.png')
    print("Done")