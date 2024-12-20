from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

dataset = 'R52'
data_path = f'data/{dataset}.txt'

with open(data_path, 'r') as f:
    labels = f.readlines()

all_labels = []

for line in labels:
    parts = line.strip().split('\t')
    id, data_split, label = parts[0], parts[1], parts[2]
    all_labels.append(label)


word_counts = Counter(all_labels)

wordcloud = WordCloud(width=800, height=400, background_color='white')

wordcloud.generate_from_frequencies(word_counts)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

plt.savefig('wordcloud.svg', format='svg', dpi=300)
plt.show()