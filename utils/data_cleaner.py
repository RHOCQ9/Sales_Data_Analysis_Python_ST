import pandas as pd

class DataCleaner:

    def __init__(self, dataframe):
        self.df = dataframe

    def remove_nulls(self):
        self.df = self.df.dropna()

    def remove_duplicates(self):
        self.df = self.df.drop_duplicates()

    def fix_data_types(self):
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])

    def normalize_columns(self):
        self.df.columns = self.df.columns.str.lower()

    def clean(self):
        self.normalize_columns()
        self.remove_nulls()
        self.remove_duplicates()
        self.fix_data_types()
        return self.df