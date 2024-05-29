import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm


class SpotifyRecommendationSystem:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)

    def preprocess_data(self):
        # Handle missing values
        self.data.dropna(inplace=True)

        # Scaling numerical features
        scaler = StandardScaler()
        numerical_features = ['popularity', 'duration_ms', 'danceability', 'energy', 'key',
                              'loudness', 'speechiness', 'acousticness', 'instrumentalness',
                              'liveness', 'valence', 'tempo']
        self.data[numerical_features] = scaler.fit_transform(self.data[numerical_features])

        # Encoding categorical features (if any)
        # Assuming 'explicit' and 'mode' are categorical features
        self.data['explicit'] = self.data['explicit'].map({True: 1, False: 0})
        self.data['mode'] = self.data['mode'].map({'major': 1, 'minor': 0})

        # Splitting the dataset into training and testing sets
        X = self.data.drop(columns=["Unnamed: 0", "track_id", "artists", "album_name", "track_name", "track_genre"])
        y = self.data["popularity"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self):
        # Train a RandomForestRegressor
        self.model = RandomForestRegressor()
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        # Evaluate the model
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        print("Mean Squared Error:", mse)

    def recommend(self, input_song_name, amount=1):
        distance = []
        input_song = self.data[self.data['track_name'].str.lower() == input_song_name.lower()].iloc[0]
        rec = self.data[self.data['track_name'].str.lower() != input_song_name.lower()]
        for _, song in tqdm(rec.iterrows()):
            d = 0
            for col in self.data.columns:
                if col not in ['track_name', 'artists', 'popularity', 'duration_ms']:
                    try:
                        d += np.absolute(float(input_song[col]) - float(song[col]))
                    except ValueError:
                        pass  # Skip non-numeric columns
            distance.append(d)

        rec['distance'] = distance
        rec = rec.sort_values('distance')
        columns = ['artists', 'track_name']  # Adjust column selection as needed
        return rec[columns][:amount]


# Usage example:
if __name__ == "__main__":
    # Initialize recommendation system
    recommendation_system = SpotifyRecommendationSystem("dataset.csv")

    # Preprocess the data
    recommendation_system.preprocess_data()

    # Train the model
    recommendation_system.train_model()

    # Evaluate the model
    recommendation_system.evaluate_model()

    # Get recommendations
    recommendations = recommendation_system.recommend("Lovers Rock", 10)
    print(recommendations)
