class SalesAnalyzer:

    def __init__(self, dataframe):
        self.df = dataframe

    def total_sales(self):
        return self.df['total_sales'].sum()

    def sales_by_product(self):
        return self.df.groupby('product')['total_sales'].sum()

    def sales_by_region(self):
        return self.df.groupby('region')['total_sales'].sum()

    def monthly_sales(self):
        self.df['month'] = self.df['date'].dt.to_period('M')
        return self.df.groupby('month')['total_sales'].sum()

    def top_products(self, n=5):
        return (
            self.df.groupby('product')['total_sales']
            .sum()
            .sort_values(ascending=False)
            .head(n)
        )

    def generate_summary(self):
        return {
            "total_sales": self.total_sales(),
            "top_products": self.top_products(),
            "sales_by_region": self.sales_by_region()
        }