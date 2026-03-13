class DatasetDetector:

    def __init__(self, dataframe):
        self.df = dataframe

    def detect_numeric(self):
        return list(self.df.select_dtypes(include=['number']).columns)

    def detect_categorical(self):
        return list(self.df.select_dtypes(include=['object']).columns)

    def detect_datetime(self):
        return list(self.df.select_dtypes(include=['datetime']).columns)

    def dataset_summary(self):

        return {
            "numeric": self.detect_numeric(),
            "categorical": self.detect_categorical(),
            "datetime": self.detect_datetime()
        }