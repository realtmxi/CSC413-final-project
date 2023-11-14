import pandas as pd
import matplotlib.pyplot as plt


csv_file_path = 'blkjckhands.csv'

df = pd.read_csv(csv_file_path)

df1 = df['sumofdeal']
value_counts = df1.value_counts()
value_counts = value_counts.sort_index()
plt.subplot(2, 2, 1)
value_counts.plot(kind='bar', edgecolor='black', figsize=(10, 6))

plt.title('Distribution of Values of Each Hand of Dealer')
plt.xlabel('Unique Values of Each Hand')
plt.ylabel('Frequency')
# plt.show()


df2 = df['sumofcards']
value_counts1 = df2.value_counts()
value_counts1 = value_counts1.sort_index()
plt.subplot(2, 2, 2)
value_counts1.plot(kind='bar', edgecolor='black', figsize=(10, 6))

plt.title('Distribution of Values of Each Hand of Player')
plt.xlabel('Unique Values of Each Hand')
plt.ylabel('Frequency')
# plt.show()


df3 = df['ply2cardsum']
value_counts2 = df3.value_counts()
value_counts2 = value_counts2.sort_index()
plt.subplot(2, 2, 3)
value_counts2.plot(kind='bar', edgecolor='black', figsize=(10, 6))

plt.title('Distribution of the Sum of First Two Cards of Player')
plt.xlabel('Unique Values of Each Sum')
plt.ylabel('Frequency')
# plt.show()


df4 = df['winloss']
plt.subplot(2, 2, 4)
value_counts3 = df4.value_counts()
value_counts3 = value_counts3.sort_index()
value_counts3.plot(kind='bar', edgecolor='black', figsize=(10, 6))

plt.title('Distribution for Results of Each Hand for Player')
plt.xlabel('Each outcome')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('bar_hand.png')
plt.show()


