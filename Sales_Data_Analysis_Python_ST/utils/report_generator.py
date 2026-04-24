import pandas as pd
import io

class ReportGenerator:
    def __init__(self, df):
        self.df = df

    def to_csv(self):
        """Convierte el DataFrame filtrado a CSV."""
        return self.df.to_csv(index=False).encode('utf-8')

    def to_summary_text(self):
        """Genera un reporte resumido en formato de texto."""
        report = "REPORTE DE VENTAS\n"
        report += "==================\n\n"
        
        if "total_sales" in self.df.columns:
            report += f"Ventas Totales: {self.df['total_sales'].sum():.2f}\n"
        
        if "quantity" in self.df.columns:
            report += f"Cantidad Total: {self.df['quantity'].sum()}\n"
        
        report += f"Número de Órdenes: {len(self.df)}\n\n"
        
        if "product" in self.df.columns and "total_sales" in self.df.columns:
            report += "Ventas por Producto:\n"
            report += self.df.groupby("product")["total_sales"].sum().to_string()
            report += "\n\n"
            
        if "region" in self.df.columns and "total_sales" in self.df.columns:
            report += "Ventas por Región:\n"
            report += self.df.groupby("region")["total_sales"].sum().to_string()
            report += "\n"
            
        return report.encode('utf-8')
