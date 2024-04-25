from diagrams import Cluster, Diagram
from diagrams.onprem.client import User, Client
from diagrams.onprem.compute import Server
from diagrams.custom import Custom
from diagrams.aws.compute import ECS
from diagrams.aws.database import ElastiCache, RDS
from diagrams.aws.network import ELB
from diagrams.aws.network import Route53

with Diagram("SkyGuide Control Pipeline", show=True):
    user = User("User")
    wifi = Custom("WiFi", "./assets/wifi_icon.png")
    tello = Custom("Tello", "./assets/tello.png")

    with Cluster("Main Thread"):
        main_thread = Custom("Frame (Camera Input)", "./frame.png")
        visual_response = Custom("Visual Response Algorithm", "./assets/algorithm.png")
        main_thread >> Custom("YOLO", "./yolo.png") >> visual_response >> Client('OpenCV Window')

    with Cluster("Worker Thread"):
        worker_thread = Server('Command')
        queue = Custom('Queue', './assets/queue.png') 
        worker_thread >> queue >> Custom('Movement', './assets/movement.png')


    user >> wifi >> tello
    tello >> wifi >> user
    visual_response >> worker_thread
    tello >> main_thread
    tello >> worker_thread