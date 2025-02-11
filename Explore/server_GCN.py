# server.py
import flwr as fl

def fit_metrics_aggregation_fn(metrics):
    """Aggregate fit metrics using a weighted average."""
    if not metrics:
        return {}
    total_examples = sum(num_examples for num_examples, _ in metrics)
    aggregated = {}
    for num_examples, client_metrics in metrics:
        for key, value in client_metrics.items():
            aggregated[key] = aggregated.get(key, 0) + value * num_examples
    for key in aggregated:
        aggregated[key] /= total_examples
    return aggregated

def evaluate_metrics_aggregation_fn(metrics):
    """Aggregate evaluation metrics using a weighted average."""
    if not metrics:
        return {}
    total_examples = sum(num_examples for num_examples, _ in metrics)
    aggregated = {}
    for num_examples, client_metrics in metrics:
        for key, value in client_metrics.items():
            aggregated[key] = aggregated.get(key, 0) + value * num_examples
    for key in aggregated:
        aggregated[key] /= total_examples
    return aggregated

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1,            # Use all available clients for training.
    fraction_evaluate=1,       # Use all available clients for evaluation.
    min_fit_clients=5,         # Adjust as needed (e.g., 10 in a larger deployment).
    min_evaluate_clients=5,
    min_available_clients=5,
    fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
)

def main():
    # Start Flower server on port 8080 for 3 rounds.
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()

# python server_GCN.py
# python clinet_GCNN.py 0 ./Fraud_dataset/partition_0.csv
# python clinet_GCNN_v2.py 0 ./Fraud_dataset/partition_0.csv
