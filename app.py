import random
from flask import Flask, request, render_template_string, redirect, url_for
import pickle
import pandas as pd
from predictors.latent_factor import LatentFactorPredictor
from recommenders import Recommender
from recommenders.auction import AuctionRecommender
from recommenders.plain import PlainRecommender
from recommenders.rating_boost import RatingBoostRecommender

app = Flask(__name__)

# Load movie id to title mapping from the MovieLens movies CSV file.
# Assumes that fetch_datasets.py has extracted the dataset into the "data" folder.
try:
    movies_df = pd.read_csv("data/ml-latest/movies.csv")
    movie_id_mapping = dict(zip(range(len(movies_df)), movies_df["title"]))
except Exception as e:
    movie_id_mapping = {}
    print("Error loading movies.csv:", e)

# Load the latent model saved in the models folder.
with open("models/latent_5", "rb") as f:
    latent: LatentFactorPredictor = pickle.load(f)

fixed_random = random.Random(0xC0FFEE)

# Create a plain recommender using the latent predictor.
latent_users = latent.p.shape[1]
latent_items = latent.q.shape[1]
plain_recommender = PlainRecommender(
    predictor=latent, users=latent_users, items=latent_items
)

payments = [
    (idx, fixed_random.random())
    for idx in fixed_random.sample(range(latent_items), k=50)
]
rating_boost_recommender = RatingBoostRecommender(
    predictor=latent,
    users=latent_users,
    items=latent_items,
    payments=payments,
    alpha=0.2,
    beta=25,
    promotion_slots=[x for x in range(0, 20, 4)],
)

bids = [
    (idx, fixed_random.randint(0, 4), fixed_random.random())
    for idx in fixed_random.sample(range(latent_items), k=50)
]
bids.sort(key=lambda x: x[1])
auction_recommender = AuctionRecommender(
    predictor=latent,
    users=latent_users,
    items=latent_items,
    bids=bids,
    alpha=0.5,
    beta=3,
    promotion_slots=[x for x in range(0, 20, 4)],
)

# Define available model choices.
# "latent" here could be used directly if it provides recommendations, and
# "plain" uses the PlainRecommender wrapper.
models: dict[str, Recommender] = {
    "plain": plain_recommender,
    "rating_boost": rating_boost_recommender,
    "auction": auction_recommender,
}


# Index page: shows a form to choose the model and user id.
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        model_choice = request.form.get("model")
        user_id = request.form.get("user_id")
        if not user_id:
            return redirect(url_for("index"))
        return redirect(url_for("predict", model_choice=model_choice, user_id=user_id))
    return render_template_string(
        """
        <h1>Recommendation Demo</h1>
        <form method="post">
          <label for="model">Choose model:</label>
          <select name="model" id="model">
            <option value="plain">Plain Recommender</option>
            <option value="rating_boost">Rating Boost Recommender</option>
            <option value="auction">Auction Based Recommender</option>
          </select>
          <br><br>
          <label for="user_id">User ID:</label>
          <input type="number" name="user_id" id="user_id" required>
          <br><br>
          <input type="submit" value="Get Recommendations">
        </form>
        """
    )


# Prediction endpoint: gets the chosen model and user id, then shows recommendations.
@app.route("/predict")
def predict():
    model_choice = request.args.get("model_choice")
    user_id = request.args.get("user_id")

    if model_choice not in models:
        return "Invalid model choice", 400

    try:
        user_id_int = int(user_id)
    except ValueError:
        return "Invalid user id", 400

    recommender = models[model_choice]
    # Get recommendations. Assumes that the recommender has a method recommend_items(user, n)
    recommended_items = recommender.recommend_items(user_id_int, 20)

    # Map recommended item ids to titles.
    recommended_titles_and_ratings = [
        (
            item[0],
            movie_id_mapping.get(item[0], f"Movie ID {item[0]}"),
            item[1],
            item[2],
        )
        for item in recommended_items
    ]

    return render_template_string(
        """
        <h1>Recommendations for User {{ user_id }}</h1>
        <p>Model used: {{ model_choice }}</p>
        {% if model_choice == 'rating_boost' %}
          Payments:
          <ul style="height: 30rem; overflow: auto">
            {% for payment in recommender.payments %}
              <li>Item {{ payment[0] }}: {{ payment[1] }}</li>
            {% endfor %}
          </ul>
        {% endif %}
        {% if model_choice == 'auction' %}
          Bids:
          <ul style="height: 30rem; overflow: auto">
            {% for bid in recommender.bids %}
              <li>Item {{ bid[0] }}, slot {{ bid[1] }}, bid {{ bid[2] }}</li>
            {% endfor %}
          </ul>
        {% endif %}
        <ul>
          {% for idx, item in items %}
            {% if idx % 4 == 0 and model_choice != 'plain' %}
                <li style="text-decoration: underline;">ID: {{ item[0] }}, {{ item[1] }}, rating: {{ item[2] }}, previous: {{ item[3] }}</li>
            {% else %}
                <li>ID: {{ item[0] }}, {{ item[1] }}, rating: {{ item[2] }}</li>
            {% endif %}
          {% endfor %}
        </ul>
        <br>
        <a href="{{ url_for('index') }}">Back</a>
        <hr>
        <h2>Get More Recommendations:</h2>
        <form method="post" action="{{ url_for('index') }}">
          <label for="model">Choose model:</label>
          <select name="model" id="model">
            <option value="plain">Plain Recommender</option>
            <option value="rating_boost">Rating Boost Recommender</option>
            <option value="auction">Auction Based Recommender</option>
          </select>
          <br><br>
          <label for="user_id">User ID:</label>
          <input type="number" name="user_id" id="user_id" required>
          <br><br>
          <input type="submit" value="Get Recommendations">
        </form>
        """,
        user_id=user_id_int,
        model_choice=model_choice,
        items=enumerate(recommended_titles_and_ratings),
        recommender=recommender,
    )


@app.route("/random-payments")
def random_payments():
    """
    Generates random bids for the paid recommender.
    """
    num_items = latent.q.shape[1]
    new_payments = [
        (idx, fixed_random.random())
        for idx in fixed_random.sample(range(num_items), k=50)
    ]
    rating_boost_recommender.payments = new_payments
    return "Random payments generated", 200


@app.route("/random-bids")
def random_bids():
    """
    Generates random bids for the auction recommender.
    """
    num_items = latent.q.shape[1]
    new_bids = [
        (idx, fixed_random.randint(0, 4), fixed_random.random())
        for idx in fixed_random.sample(range(num_items), k=50)
    ]
    auction_recommender.bids = new_bids
    return "Random bids generated", 200


if __name__ == "__main__":
    app.run(debug=True)
