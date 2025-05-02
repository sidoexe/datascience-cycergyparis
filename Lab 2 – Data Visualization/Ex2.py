import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("company_sales_data.csv")

# Plot 1: Total Profit per Month
# Reason i choosed line plot because it's a comparison over time
# i analyzed the data and found that the total profit is increasing over time, the most profitable period is the end of the year
data.plot(x='month_number', y='total_profit', kind='line', figsize=(10, 5))
plt.show()

# Plot 2: Sales of Different Products per Month
# Reason i choosed bar plot because it's a comparison among items
# i analyzed the data and found that the most sold product is bathingsoap, it's by far the best selling product with huge difference between it and the second best selling product which is the toothpaste
products = ['facecream', 'facewash', 'toothpaste', 'bathingsoap', 'shampoo', 'moisturizer']
data.plot(x='month_number', y=products, kind='bar', figsize=(10, 5))
plt.show()

# Plot 3: Face Cream vs. Toothpaste Sales per Month
# Reason i choosed line plot because it's a comparison over time
# i analyzed the data and first found that the toothpaste has more sales than the facecream, and in the month 9 when the facecream sales decreased the toothpaste sales increased simultaneously which can be a sign
data.plot(x='month_number', y=['facecream', 'toothpaste'], kind='line', figsize=(10, 5))
plt.show()

# # Plot 4: Distribution of Total Profits
# Reason i choosed histogram because it's a distribution of data
data.plot(kind='hist', figsize=(10, 5))
plt.show()

# # Plot 5: Product Sales and Cumulative Sales
# Reason i choosed area plot because it's composition of data over time
# i analyzed the data and found that the total units sold is increasing over time, and the mosturizer is the most sold in terms of total units 
data.plot(x='month_number', y=['facecream', 'facewash', 'toothpaste', 'bathingsoap', 'shampoo', 'moisturizer', 'total_units'], kind='area', figsize=(10, 5))
plt.show()