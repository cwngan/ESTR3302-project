import pandas as pd
from random import sample
from scipy.sparse import csr_matrix


class Dataset:
    def __init__(self, full: bool = False):
        print("Reading CSV files...")
        if full:
            # Read the ratings and movies CSV (WARNING: FULL DATASET)
            self.ratings_df = pd.read_csv("data/ml-latest/ratings.csv")
            self.movies_df = pd.read_csv("data/ml-latest/movies.csv")
        else:
            # Read the ratings and movies CSV
            self.ratings_df = pd.read_csv("data/ml-latest-small/ratings.csv")
            self.movies_df = pd.read_csv("data/ml-latest-small/movies.csv")

        # Convert the CSV into a user ratings table
        # Create a dense matrix where each row represents a user and each column a movie.
        # Missing ratings are filled with 0.
        print("Pivotting data...")
        self.user_ids = sorted(self.ratings_df["userId"].unique())
        self.movie_ids = sorted(self.ratings_df["movieId"].unique())
        self.user_id_map = {uid: i for i, uid in enumerate(self.user_ids)}
        self.movie_id_map = {mid: j for j, mid in enumerate(self.movie_ids)}

        rows = self.ratings_df["userId"].map(self.user_id_map).values
        cols = self.ratings_df["movieId"].map(self.movie_id_map).values
        data = self.ratings_df["rating"].values

        self.user_ratings = csr_matrix(
            (data, (rows, cols)), shape=(len(self.user_ids), len(self.movie_ids))
        )
        self.movie_id_mappings = self.movies_df["movieId"].to_list()

        print(
            "User ratings table created with dimensions:",
            self.user_ratings.shape[0],
            "rows x",
            self.user_ratings.shape[1],
            "columns",
        )

        print("Making test sets...")
        # Get all indices with an existing (non zero) rating
        valid_entries = list(zip(*self.user_ratings.nonzero()))
        shuffuled_valid_entries = sample(valid_entries, k=len(valid_entries))
        test_set_size = int(len(valid_entries) * 0.2)

        # Randomly select test_set_size indices from the valid entries
        self.test_set = shuffuled_valid_entries[:test_set_size]
        self.training_set = shuffuled_valid_entries[test_set_size:]
        print("Test set created with size:", len(self.test_set))
        print("Done.")
