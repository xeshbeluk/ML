# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

class GloVe_Embedder:
    def __init__(self, path):
        self.embedding_dict = {}
        self.embedding_array = []
        self.unk_emb = 0
        # Adapted from https://stackoverflow.com/questions/37793118/load-pretrained-GloVe-vectors-in-python
        with open(path,'r') as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                self.embedding_dict[word] = embedding
                self.embedding_array.append(embedding.tolist())
        self.embedding_array = np.array(self.embedding_array)
        self.embedding_dim = len(self.embedding_array[0])
        self.vocab_size = len(self.embedding_array)
        self.unk_emb = np.zeros(self.embedding_dim)

    # Check if the provided embedding is the unknown embedding.
    def is_unk_embed(self, embed):
        return np.sum((embed - self.unk_emb) ** 2) < 1e-7

    # Check if the provided string is in the vocabulary.
    def token_in_vocab(self, x):
        if x in self.embedding_dict and not self.is_unk_embed(self.embedding_dict[x]):
            return True
        return False

    # Returns the embedding for a single string and prints a warning if
    # the string is unknown to the vocabulary.
    #
    # If indicate_unk is set to True, the return type will be a tuple of
    # (numpy array, bool) with the bool indicating whether the returned
    # embedding is the unknown embedding.
    #
    # If warn_unk is set to False, the method will no longer print warnings
    # when used on unknown strings.
    def embed_str(self, x, indicate_unk = False, warn_unk = True):
        if self.token_in_vocab(x):
            if indicate_unk:
                return (self.embedding_dict[x], False)
            else:
                return self.embedding_dict[x]
        else:
            if warn_unk:
                    print("Warning: provided word is not part of the vocabulary!")
            if indicate_unk:
                return (self.unk_emb, True)
            else:
                return self.unk_emb

    # Returns an array containing the embeddings of each vocabulary token in the provided list.
    #
    # If include_unk is set to False, the returned list will not include any unknown embeddings.
    def embed_list(self, x, include_unk = True):
        if include_unk:
            embeds = [self.embed_str(word, warn_unk = False).tolist() for word in x]
        else:
            embeds_with_unk = [self.embed_str(word, indicate_unk=True, warn_unk = False) for word in x]
            embeds = [e[0].tolist() for e in embeds_with_unk if not e[1]]
            if len(embeds) == 0:
                print("No known words in input:" + str(x))
                embeds = [self.unk_emb.tolist()]
        return np.array(embeds)

    # Finds the vocab words associated with the k nearest embeddings of the provided word.
    # Can also accept an embedding vector in place of a string word.
    # Return type is a nested list where each entry is a word in the vocab followed by its
    # distance from whatever word was provided as an argument.
    def find_k_nearest(self, word, k, warn_about_unks = True):
        if type(word) == str:
            word_embedding, is_unk = self.embed_str(word, indicate_unk = True)
        else:
            word_embedding = word
            is_unk = False
        if is_unk and warn_about_unks:
            print("Warning: provided word is not part of the vocabulary!")

        all_distances = np.sum((self.embedding_array - word_embedding) ** 2, axis = 1) ** 0.5
        distance_vocab_index = [[w, round(d, 5)] for w,d,i in zip(self.embedding_dict.keys(), all_distances, range(len(all_distances)))]
        distance_vocab_index = sorted(distance_vocab_index, key = lambda x: x[1], reverse = False)
        return distance_vocab_index[:k]

    def save_to_file(self, path):
        with open(path, 'w') as f:
            for k in self.embedding_dict.keys():
                embedding_str = " ".join([str(round(s, 5)) for s in self.embedding_dict[k].tolist()])
                string = k + " " + embedding_str
                f.write(string + "\n")

# Your code goes here
ge = GloVe_Embedder(EMBEDDING_PATH)
#
# Embed single word via:
embed = ge.embed_str('flight')
#
# Embed a list of words via:
wordss = ['flight', 'awesome', 'terrible', 'help', 'late']
embeds = ge.embed_list(wordss)
print(ge.find_k_nearest('flight', 30))

# Find k nearest neighbors of word via:
all_150_words = pd.DataFrame()
for i in wordss:
  res = pd.DataFrame(ge.find_k_nearest(i, 30), columns=['word', 'weight'])
  res['seed_word'] = i

  all_150_words = pd.concat([all_150_words, res], ignore_index=True)
print(all_150_words)

# Save vocabulary to file via:
ge.save_to_file('/content/gdrive/My Drive/AI534/all_words.csv')

embeds_for_150 = ge.embed_list(all_150_words['word'])

seed_words = all_150_words['seed_word']

scaler = StandardScaler()
embeddings_standardized = scaler.fit_transform(embeds_for_150)
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings_standardized)
all_150_words_2d = pd.DataFrame(data=embeddings_2d, columns=['PC1', 'PC2'])

all_150_words_2d['seed_word'] = seed_words

# Display the resulting DataFrame
print(all_150_words_2d)

combined_df = all_150_words_2d

# Plotting
plt.figure(figsize=(15, 10))
colors = {'flight': 'red', 'awesome': 'blue', 'terrible': 'green', 'help': 'orange', 'late': 'purple'}
for seed_word in combined_df['seed_word'].unique():
    subset = combined_df[combined_df['seed_word'] == seed_word]
    plt.scatter(subset['PC1'], subset['PC2'], label=seed_word, color=colors[seed_word])

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA of Word Embeddings')
plt.legend()
plt.show()

"""The colors do seperate the data into 5 clusters. However it does look more naturally seperated into 4 clusters. With out the colors it would be difficult to identify 5 clear seperate groups. The green and blue dots (awesome and terrible) appear to be the most mixed together. This might be because both words are adjectives describing how good/bad an event wa. Overall This there was a lot of data reducation. PCA took the data from alot of dimentions down to just two (for visuallisation). I think that if we had a few more dimentions the data might be more seperated even though we could not visualize it well.

"""

perplexities = [5, 10, 20, 30, 40, 50]
plt.figure(figsize=(15, 10))

for i, perplexity in enumerate(perplexities):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_standardized)
    all_150_words_2d = pd.DataFrame(data=embeddings_2d, columns=['Dimension 1', 'Dimension 2'])
    all_150_words_2d['seed_word'] = seed_words
    combined_df = all_150_words_2d

    plt.subplot(2, 3, i + 1)
    colors = {'flight': 'red', 'awesome': 'blue', 'terrible': 'green', 'help': 'orange', 'late': 'purple'}
    for seed_word in combined_df['seed_word'].unique():
        subset = combined_df[combined_df['seed_word'] == seed_word]
        plt.scatter(subset['Dimension 1'], subset['Dimension 2'], label=seed_word, color=colors[seed_word])
    plt.title(f't-SNE with Perplexity={perplexity}')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

plt.tight_layout()
plt.show()

"""While the data does not appear to be in 5 clear clusters, we can see that as perplexity increased the colors were more grouped together. Higher perplexity in this case ment less overlap of the colors. I am not 100% sure what the perplexity hyperparameter is doing in the model, but I would want to make sure that is it not overfitting to the data if it got too high. I think it is okay in our example here but that is something I would look out for."""

kmeans_objectives = []

for k in range(2, 21):
    kmeans = KMeans(n_clusters=k,n_init=10)
    kmeans.fit(embeds_for_150)
    kmeans_objectives.append(kmeans.inertia_)

kmeans_results = pd.DataFrame({'k': range(2, 21), 'inertia': kmeans_objectives})
kmeans_results

plt.plot(kmeans_results['k'], kmeans_results['inertia'], marker='o')
plt.title('Dmitrys and Joshua K-Means kalapolooza')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.show()

"""The "elbow" of this graph is not very clear, but it does look like 5 is where that rate of change really does shift. This means that inertia reduces a great amount between 2 and 3, 3 and 4, 4 and 5, but the amount of change after adding just one more cluster after that because very small. This indicates to us that 5 might be the appropriate about of clusters. If we just wanted to minimize inertia, we would just pick the highest number of clusters, but we don't want to do that. We want to get the right about of clusters to accuratly depict the data and this graph indicates that 5 might be a good place to do that."""

true_labels = all_150_words['seed_word']\
      #a function to calculate purity
def calculate_purity(true_labels, predicted_labels):
    contingency_matrix = np.zeros((len(np.unique(true_labels)), len(np.unique(predicted_labels))))

    for i, true_label in enumerate(np.unique(true_labels)):
        true_indices = (true_labels == true_label)
        predicted_cluster = np.argmax(np.bincount(predicted_labels[true_indices]))
        contingency_matrix[i, predicted_cluster] = np.sum(true_indices)

    return np.sum(np.amax(contingency_matrix, axis=0)) / len(true_labels)

for k in range(2, 21):
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=k, n_init=100)
    predicted_labels = kmeans.fit_predict(embeds_for_150)  # Use the embeddings_array from the previous example

    # Evaluate clustering solution using different metrics
    purity = calculate_purity(true_labels, predicted_labels)
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    print(f'For k={k}: Purity = {purity:.4f}, ARI = {ari:.4f}, NMI = {nmi:.4f}')

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
import nltk

train_data = pd.read_csv(train_path)
val_data = pd.read_csv(val_path)
X_train_text = train_data['text']
y_train = train_data['sentiment']
X_val_text = val_data['text']
y_val = val_data['sentiment']

def tokenize_tweet(tweet):
    return word_tokenize(tweet)

X_train_words = X_train_text.apply(tokenize_tweet)
X_val_words = X_val_text.apply(tokenize_tweet)

embeds_train = []

for words in X_train_words:
    embeddings = ge.embed_list(words)
    avg_embedding = np.mean(embeddings, axis=0)
    embeds_train.append(avg_embedding)

embeds_train = np.array(embeds_train)

embeds_val = []

for words in X_val_words:
    embeddings = ge.embed_list(words)
    avg_embedding = np.mean(embeddings, axis=0)
    embeds_val.append(avg_embedding)

embeds_val = np.array(embeds_val)


# Initialize Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, min_samples_split=5)
rf_model.fit(embeds_train, y_train)

# Predictions on the validation set
y_pred_val = rf_model.predict(embeds_val)

# Evaluate the model
print(classification_report(y_val, y_pred_val))

"""Basically we just took each tweet, then seperated out all of the words and got the average embeddings for each tweet. Then we fit a random forest using the data and selected some hyperparameters that worked well and we thought was not overfitting. The accuracy is decent (85%) and the precision is pretty good. The recall struggled with positive tweets.  Obiously more robust weighting on the embeddings, fine tuning the hyperparameters, and more feature selection would help impove this type of model."""
