import pandas as pd


class DataLoader:
    def __init__(self, dataset_path: str, policy_path: str):
        self.dataset = pd.read_csv(dataset_path)
        with open(policy_path, "r", encoding="utf-8") as f:
            self.policy = f.read()

    def get_subset(self, max_clients: int, start_idx: int = 0) -> pd.DataFrame:
        return self.dataset.iloc[start_idx : start_idx + max_clients]

    def get_head(self, max_clients: int) -> pd.DataFrame:
        return self.dataset.head(max_clients)

    def drop_indices(self, indices: set, max_clients: int) -> pd.DataFrame:
        return self.dataset.drop(index=indices).head(max_clients)

    def format_client(self, idx: int, row: pd.Series) -> str:
        return (
            f"Client {idx + 1}: Travel Class={row['travel_class']}, "
            f"Age Category={row['age_category']}, Luggage={row['luggages']}"
        )
