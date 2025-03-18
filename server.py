import flwr as fl
from flwr.server import ServerConfig

# Define a strategy (e.g., Federated Averaging)
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Use all available clients for training
    fraction_evaluate=1.0
)

# Start the Flower server
fl.server.start_server(
    #server_address="0.0.0.0:8080",  # Server listens on all interfaces
    #server_address="localhost:8080",
    config=ServerConfig(num_rounds=3),  # Number of federated learning rounds
    strategy=strategy
)

#def server_fn(context: Context) -> ServerAppComponents:
#    config = ServerConfig(num_rounds=5)
#    return ServerAppComponents(strategy=strategy, config=config)

# Create the ServerApp
#server = ServerApp(server_fn=server_fn)