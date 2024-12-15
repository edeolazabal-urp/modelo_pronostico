import pandas as pd

# Load the CSV file to analyze its structure
file_path = '/mnt/data/ventas_mineria_actualizado.csv'
data = pd.read_csv(file_path)

# Display the first few rows and the column information to understand the structure
data_info = data.info()
data_head = data.head()

data_info, data_head
