import flwr as fl


def weighted_average(metrics):
    # A helper to average evaluation metrics across clients.
    accuracy = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracy) / sum(examples)}


# We use FedAvg as the strategy, aggregating all parameters (front + back).
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1,
    fraction_evaluate=0.6,
    min_fit_clients=5,
    min_evaluate_clients=5,
    min_available_clients=10,
    evaluate_metrics_aggregation_fn=weighted_average,
)

if __name__ == "__main__":
    # Start Flower server on port 8080 for 10 rounds.
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )
