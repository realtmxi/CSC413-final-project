import pandas as pd
import matplotlib.pyplot as plt


csv_file_path = 'blkjckhands.csv'

df = pd.read_csv(csv_file_path)

df2 = df['sumofcards']

value_counts1 = df2.value_counts()

value_counts1 = value_counts1.sort_index()

value_counts1.plot(kind='bar', edgecolor='black', figsize=(10, 6))

plt.title('Distribution of Values of Each Hand')
plt.xlabel('Unique Values of Each Hand')
plt.ylabel('Frequency')

plt.show()
