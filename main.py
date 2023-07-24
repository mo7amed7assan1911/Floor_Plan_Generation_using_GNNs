from utils import *
from model import *
from upload import *

def get_info():
    """
        Function to return:
            - Boundary: "POLYGON ((105.44810944559121 78 ....
            - front_door: "POLYGON  (105.44810944559121 78 ....
            - room_centroids: [(81, 105), (55, 151), (134, 105)]
            - bathroom_centroids: [(81, 105), (55, 151), (134, 105)]
            - kitchen_centroids: [(81, 105), (55, 151), (134, 105)]
    """
    boundary_wkt = "POLYGON ((25.599999999999994 65.32413793103447, 200.38620689655173 65.32413793103447, 200.38620689655173 75.91724137931033, 230.4 75.91724137931033, 230.4 190.67586206896553, 67.97241379310344 190.67586206896553, 67.97241379310344 176.55172413793102, 25.599999999999994 176.55172413793102, 25.599999999999994 65.32413793103447))"
    
    front_door_wkt = "POLYGON ((38.436315932155225 179.69850789734912, 63.610586007499926 179.69850789734912, 63.610586007499926 176.55172413793102, 38.436315932155225 176.55172413793102, 38.436315932155225 179.69850789734912))"
    
    # Data of the inner rooms or bathrooms
    room_centroids  = [(201, 163), (193, 106)]
    bathroom_centroids = [(91, 91), (52, 95)]
    kitchen_centroids = [(137, 89)]
    
    # boundary_wkt = input("Enter the boundary as str: ")
    # front_door_wkt = input("Enter the front door as str: ")
    # room_centroids = input("Enter the room centroids as list of tuples: ")
    # bathroom_centroids = input("Enter the bathroom centroids as list of tuples: ")
    # kitchen_centroids = input("Enter the kitchen centroids as list of tuples: ")
    
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

    return G, B, G_not_normalized, B_not_normalized, Boundary, front_door, B_n, G_n


    
def Run(Boundary, front_door, room_centroids, bathroom_centroids, kitchen_centroids):
    # Get the data
    # Boundary, front_door, room_centroids, bathroom_centroids, kitchen_centroids = get_info()
    
    # ========================================================================
    # Preprocessing
    G, B, G_not_normalized, B_not_normalized, Boundary_as_polygon, front_door_as_polygon, B_n, G_n = preProcessing_toGraphs(Boundary, front_door, room_centroids, bathroom_centroids, kitchen_centroids)
    the_door = Point(B_not_normalized.x[-1][1:].detach().cpu().numpy()).buffer(3)
    
    # geeing the corresponding graph for the inputs of the user
    
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
    
    #=========================================================================
    # Saving the output & Updating to the firebase
    # unique_name = str(uuid.uuid4())
    # if not os.path.exists("./Outputs"):
    #     os.mkdir("Outputs/" + unique_name)
    # plt.savefig("Outputs/" + '/Output.png')
    # image_url = upload_to_firebase(unique_name)
    # print(image_url)
    # print("Done")
    
    
    path = "Outputs/model_output.png"
    plt.savefig(path)
    plt.close()
    return path, B_n, G_n


if __name__ == '__main__':
    Run()