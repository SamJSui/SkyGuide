from diagrams import Cluster, Diagram
from diagrams.onprem.client import User, Client
from diagrams.onprem.compute import Server
from diagrams.custom import Custom

graph_attr = {
    "fontsize": "32",
    "bgcolor": "transparent"
}

edge_attr = {
    "color": "black",
    "penwidth": "2.0",
    "fontsize": "16",
    "fontcolor": "black"
}

node_attr = {
    "style": "filled",
    "fillcolor": "lightblue",
    "fontsize": "16",
    "fontcolor": "black"  # Change the font color to white for better visibility
}

node_attr_frame = {
    "style": "filled",
    "fillcolor": "lightblue",
    "fontsize": "16",
    "fontcolor": "white"  # Change the font color to white for better visibility
}

with Diagram(
    "SkyGuide Control Pipeline", 
    show=False, 
    graph_attr=graph_attr,
    edge_attr=edge_attr,
    node_attr=node_attr
):
    user = User("User")
    wifi = Custom("WiFi", "./assets/wifi_icon.png")
    tello = Custom("Tello", "./assets/tello.png")

    with Cluster("Main Thread"):
        main_thread = Custom("Frame", "./assets/frame.png", **node_attr_frame)
        visual_response = Custom("Visual Response", "./assets/algorithm.png")
        main_thread >> Custom("YOLO", "./assets/yolo.png", **node_attr_frame) >> visual_response >> Client('OpenCV Window')

    with Cluster("Worker Thread"):
        worker_thread = Server('Command')
        queue = Custom('Queue', './assets/queue.png') 
        worker_thread >> queue >> Custom('Movement', './assets/movement.png')


    user >> wifi >> tello
    tello >> wifi >> user
    visual_response >> worker_thread
    tello >> main_thread
    tello >> worker_thread
