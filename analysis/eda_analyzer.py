import pandas as pd
import plotly.express as px

class EDAAnalyzer:

    def __init__(self, dataframe):
        self.df = dataframe

    def basic_info(self):
        return {
            "rows": self.df.shape[0],
            "columns": self.df.shape[1],
            "column_names": list(self.df.columns)
        }

    def summary_statistics(self):
        return self.df.describe()

    def missing_values(self):
        return self.df.isnull().sum()

    def correlation_matrix(self):
        numeric_df = self.df.select_dtypes(include=['number'])
        return numeric_df.corr()

    def plot_distribution(self, column):
        fig = px.histogram(self.df, x=column)
        return fig

    def plot_correlation_heatmap(self):

        corr = self.correlation_matrix()

        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Matriz de correlación"
        )

        return fig