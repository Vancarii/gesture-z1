import websockets
import asyncio
import json
import logging

HOST = "0.0.0.0"
PORT = 8765
logging.basicConfig(level=logging.INFO)

class LeapServer:
    def __init__(self):
        self.clients = set()
        self.loop = None
    
    async def register(self, websocket):
        self.clients.add(websocket)
        logging.info(f"New client connected: {websocket.remote_address}")
        try:
            async for message in websocket:
                pass
        finally:
            self.clients.discard(websocket)
            logging.info(f"Client disconnected: {websocket.remote_address}")
    
    async def broadcast(self, cmddata):
        if not self.clients:
            return
        
        message = json.dumps(cmddata)
        await asyncio.gather(*(c.send(message) for c in self.clients))
    
    async def start(self):
        logging.info(f"Starting data server on ws://{HOST}:{PORT}")
        async with websockets.serve(self.register, HOST, PORT):
            await asyncio.Future()
    
    def background(self):
        pass

server = LeapServer()

if __name__ == "__main__":
    asyncio.run(server.start())
