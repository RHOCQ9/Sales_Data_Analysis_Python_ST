import pandas as pd

class DataLoader:

    def __init__(self):
        self.df = None

    def load_csv(self, path):
        self.df = pd.read_csv(path)
        return self.df

    def load_excel(self, path):
        self.df = pd.read_excel(path)
        return self.df

    def preview_data(self, rows=5):
        return self.df.head(rows)

    def get_dataframe(self):
        return self.df