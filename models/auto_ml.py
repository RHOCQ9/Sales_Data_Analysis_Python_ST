from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

class AutoML:

    def __init__(self, dataframe):
        self.df = dataframe

    def train_regression(self, target, features):

        X = self.df[features]
        y = self.df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        return {
            "mae": mae,
            "r2": r2,
            "predictions": predictions,
            "real": y_test
        }