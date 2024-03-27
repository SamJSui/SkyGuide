import asyncio
from tello_asyncio import Tello

class Drone:

    def __init__(self):
        self.tello = Tello()
        self.tello.video