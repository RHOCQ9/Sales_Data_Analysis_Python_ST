import plotly.express as px

class AutoVisualizer:

    def __init__(self, dataframe):
        self.df = dataframe

    def numeric_distribution(self, column):

        fig = px.histogram(
            self.df,
            x=column,
            title=f"Distribución de {column}"
        )

        return fig

    def categorical_counts(self, column):

        counts = (
            self.df[column]
            .value_counts()
            .reset_index()
        )

        # Renombrar columnas correctamente
        counts.columns = [column, "count"]

        fig = px.bar(
            counts,
            x=column,
            y="count",
            title=f"Conteo de {column}"
        )

        return fig

    def correlation_heatmap(self):

        corr = self.df.select_dtypes(include='number').corr()

        fig = px.imshow(
            corr,
            text_auto=True,
            title="Matriz de correlación"
        )

        return fig