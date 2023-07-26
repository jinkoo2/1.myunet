import asyncio
import websockets

url = 'ws://roweb3.uhmc.sbuh.stonybrook.edu:5000'

async def connect():
    async with websockets.connect(url) as websocket:
        while True:
            # message = input("Enter your message: ")
            
            # print('sending...', message)
            # await websocket.send(message)

            print('waitingd a command...')
            response = await websocket.recv()
            
            print("Received command from server:", response)

            print("performing the task")

            print("send the result back")

            await websocket.send("job done")

asyncio.get_event_loop().run_until_complete(connect())
