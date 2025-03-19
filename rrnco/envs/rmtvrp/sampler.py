import numpy as np


class Real_World_Sampler:
    def __init__(self):
        pass

    def sample(
        self,
        data,
        batch: int,
        num_sample: int,
        loc_dist: str = "uniform",
        num_cluster: int = 5,
    ):
        """
        Samples num_sample items for each batch from the dataset.

        Args:
            batch (int): Number of batches.
            num_sample (int): Number of samples to draw per batch.
            loc_dist (str): Distribution of the locations. Options: 'uniform', 'single_cluster',
                'multiplt_cluster', 'mixed'. Note that 'mixed' here means a mix of 'single_cluster'
                and 'uniform' location sampling methods.
            num_cluster (int): Number of clusters to sample, for 'multiplt_cluster' location
                sample method.

        Returns:
            dict: A dictionary containing sampled data for all batches.
        """
        if batch <= 0 or num_sample <= 0:
            raise ValueError("batch and num_sample must be positive integers.")

        # Ensure num_sample does not exceed the size of the dataset
        data_keys = list(data.keys())
        data_length = len(data["points"])  # Assuming all keys have compatible lengths
        if num_sample > data_length:
            raise ValueError(
                f"num_sample ({num_sample}) exceeds the available data size ({data_length})."
            )
        if data["distance"].max() > 1e5:
            outlier_indices = np.where(data["distance"] > 1e5)
            for i in range(data_length):
                num_problem_points = (outlier_indices[0] == i).sum()
                if num_problem_points > 0 and num_problem_points < data_length // 2:
                    problem_indices_r = outlier_indices[1][outlier_indices[0] == i]
                    break

            for i in range(data_length):
                num_problem_points = (outlier_indices[1] == i).sum()
                if num_problem_points < data_length // 2:
                    problem_indices_c = outlier_indices[0][outlier_indices[1] == i]
                    break
            problem_indices = np.concatenate([problem_indices_r, problem_indices_c])
            new_indices = np.delete(np.arange(data_length), problem_indices)
            points = data["points"][new_indices]
            distance = data["distance"][new_indices][:, new_indices]
            duration = data["duration"][new_indices][:, new_indices]
            data_length = len(points)
            data = {"points": points, "distance": distance, "duration": duration}
        # Randomly sample indices without replacement for each batch
        if loc_dist == "uniform":
            indices = self.uniform_sample(batch, data_length, num_sample)
        elif loc_dist == "single_cluster":
            indices = self.single_cluster_sample(
                data["points"], batch, data_length, num_sample
            )
        elif loc_dist == "multiple_cluster":
            indices = self.multiple_cluster_sample(
                data["points"], batch, data_length, num_sample, num_cluster
            )
        elif loc_dist == "mixed":
            indices = self.mixed_sample(data["points"], batch, data_length, num_sample)
        else:
            raise ValueError(f"Invalid loc_dist: {loc_dist}")

        # Create a dictionary with sampled data for all batches
        sampled_data = {}
        for key in data_keys:
            if key == "distance" or key == "duration":
                # For 'distance', create (batch, num_sample, num_sample) shape by indexing twice
                row_indices = indices[:, :, None]  # Shape: (batch, num_sample, 1)
                col_indices = indices[:, None, :]  # Shape: (batch, 1, num_sample)

                # Index the 2D distance matrix or duration matrix
                sampled_data[f"{key}_matrix"] = data[key][
                    row_indices, col_indices
                ]  # Result: (batch, num_sample, num_sample)
                # sampled_data[key] = np.array([data[key][idx][:, idx] for idx in indices])
            else:
                # For 'points', create (batch, num_sample, 2) shape
                sampled_data[key] = data[key][indices]

        return sampled_data

    def uniform_sample(self, batch, data_length, num_sample):
        indices = np.array(
            [
                np.random.choice(data_length, num_sample, replace=False)
                for _ in range(batch)
            ]
        )
        return indices

    def single_cluster_sample(self, points, batch, data_length, num_sample):
        # Random sample a center point
        center = points[np.random.choice(data_length)]
        indices = np.argsort(np.linalg.norm(points - center, axis=1))
        return indices[:num_sample]

    def multiple_cluster_sample(
        self, points, batch, data_length, num_sample, num_cluster
    ):
        # Random sample num_cluster center points
        centers = points[np.random.choice(data_length, num_cluster)]
        num_sample_per_cluster = num_sample // num_cluster
        for i in range(num_cluster):
            indices = np.argsort(np.linalg.norm(points - centers[i], axis=1))
            if i == 0:
                sampled_indices = indices[:num_sample_per_cluster]
            else:
                # Calculate the number of repeated indices
                num_repeated = num_sample_per_cluster - len(
                    np.intersect1d(sampled_indices, indices[:num_sample_per_cluster])
                )
                sampled_indices = np.concatenate(
                    (sampled_indices, indices[: num_sample_per_cluster + num_repeated])
                )
        # Remove repeat indices
        sampled_indices = np.unique(sampled_indices)
        return sampled_indices

    def mixed_sample(self, points, batch, data_length, num_sample):
        # Randomly sample some nodes
        random_indices = self.uniform_sample(batch, data_length, num_sample)
        # Cluster sample some nodes
        cluster_indices = self.single_cluster_sample(
            points, batch, data_length, num_sample
        )
        # Randomly combine the two sets of indices
        indices = np.empty((batch, num_sample), dtype=int)
        for i in range(batch):
            indices[i] = np.random.choice(
                np.concatenate((random_indices[i], cluster_indices)),
                num_sample,
                replace=False,
            )
        return indices


# if __name__ == '__main__':
#     sampler = Real_World_RCVRP_Sampler(city="daejeon")
#     sampled_data = sampler.sample(batch=2, num_sample=5)
#     print(sampled_data.keys())
#     print(sampled_data['points'].shape)
#     print(sampled_data['distance'].shape)
#     print(sampled_data["distance"])
#     print(sampled_data["points"])
