# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import seaborn as sns

"""Let's load the data."""

train_data = pd.read_csv(train_path)
val_data = pd.read_csv(val_path)

negative = train_data[train_data['sentiment'] == 0]['text']
positive = train_data[train_data['sentiment'] == 1]['text']

vectorizer = TfidfVectorizer(stop_words='english')
#All tweets
tfidf_matrix_all = vectorizer.fit_transform(train_data['text'])
feature_names_all = vectorizer.get_feature_names_out()

#Positive Tweets
tfidf_matrix_positive = vectorizer.transform(positive)
feature_names_positive = vectorizer.get_feature_names_out()

#negative tweets
tfidf_matrix_negative = vectorizer.transform(negative)
feature_names_negative = vectorizer.get_feature_names_out()

# Find words used in both positive and negative responses
positive_words = [feature_names_all[col] for col in tfidf_matrix_positive.nonzero()[1]]
negative_words = [feature_names_all[col] for col in tfidf_matrix_negative.nonzero()[1]]

# Find words in both positive and negative responses
common_words = set(positive_words).intersection(set(negative_words))

# Filter feature names for positive and negative responses
positive_words = list(set(positive_words) - common_words)
negative_words = list(set(negative_words) - common_words)

# Print results
print("Words in Positive Responses:")
print(positive_words)

print("\nWords in Negative Responses:")
print(negative_words)

print("\nWords in Both Positive and Negative Responses:")
print(list(common_words))

"""1) Positive responses do not usually contain a lot of practical suggestions since people are expected to leave feedback only if they are dissatisfied with the qulaity of service - hence, most of positive words are adjectives; Negative responses vary from direct insults to constructive feedback on how to improve particular aspects of the airline operating processes; Common respones are generally related to innate features or characterisitcs of an object a user is giving a review about.

2) The sign of the weights indicates the direction of influence, and the magnitude reflects the strength of that influence in the classification decision. We would expect verbs, nouns and prepositions to have a higher weight if the sign is negative, and adjectives and adverbs to have a higher weight if the sign is positive.  
"""

X_train = train_data['text']
y_train = train_data['sentiment']

# Separate features (X_val) and target variable (y_val) for validation
X_val = val_data['text']
y_val = val_data['sentiment']

# Use TfidfVectorizer to convert text data to TF-IDF features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# Define the SVM model
svm_model = SVC(kernel='linear')

# Define the hyperparameter grid for grid search
param_grid = {'C': [10**i for i in range(-3, 4)]}

# Perform grid search with cross-validation based on validation accuracy
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', return_train_score = True)
grid_search.fit(X_train_tfidf, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
best_c = best_params['C']

# Train the SVM model with the best hyperparameters on the entire training set
final_svm_model = SVC(kernel='linear', C=best_c)
final_svm_model.fit(X_train_tfidf, y_train)

# Record the total number of support vectors
total_support_vectors = final_svm_model.n_support_.sum()

# Validate the model on the validation set
y_train_pred = final_svm_model.predict(X_train_tfidf)
y_val_pred = final_svm_model.predict(X_val_tfidf)

# Evaluate the validation accuracy
train_accuracy = accuracy_score(y_train,y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Best C: {best_c}")
print(f"Total Support Vectors: {total_support_vectors}")
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

results = pd.DataFrame(grid_search.cv_results_)
print(results)

# Plot of training and validation accuracy vs log(C)
plt.figure(figsize=(10, 6))
plt.semilogx(param_grid['C'], results['mean_test_score'], label='Validation Accuracy', marker='o')
plt.semilogx(param_grid['C'], results['mean_train_score'], label='Train Accuracy', marker='o')

plt.title('Accuracy vs. C')
plt.xlabel('C (log scale)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

"""1) As C increases, there's a risk of overfitting, and the model may not generalize well to unseen data. Therefore, validation performance might initially improve as the model becomes more flexible, but after a certain point, it may degrade due to overfitting.

2) As C increases, the model becomes less regularized, and it tends to fit the training data more closely.The decision boundary may become more flexible, leading to more support vectors being required to define the decision boundary accurately.
"""

# Boundry Expansion
# Copy code but change grid search
param_grid = {'C': [10**i for i in range(-5, 6)]}

grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', return_train_score = True)
grid_search.fit(X_train_tfidf, y_train)

best_params = grid_search.best_params_
best_c = best_params['C']

final_svm_model = SVC(kernel='linear', C=best_c)
final_svm_model.fit(X_train_tfidf, y_train)

total_support_vectors = final_svm_model.n_support_.sum()

y_train_pred = final_svm_model.predict(X_train_tfidf)
y_val_pred = final_svm_model.predict(X_val_tfidf)

train_accuracy = accuracy_score(y_train,y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Best C: {best_c}")
print(f"Total Support Vectors: {total_support_vectors}")
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

## Grid Refinement for Central Optima (1)

# Copy code but change grid search
param_grid = {'C': [.5,1,1.5,2,3,5]}

grid_search = GridSearchCV(svm_model, param_grid, cv=3, scoring='accuracy', return_train_score = True)
grid_search.fit(X_train_tfidf, y_train)

best_params = grid_search.best_params_
best_c = best_params['C']

final_svm_model = SVC(kernel='linear', C=best_c)
final_svm_model.fit(X_train_tfidf, y_train)

total_support_vectors = final_svm_model.n_support_.sum()

y_train_pred = final_svm_model.predict(X_train_tfidf)
y_val_pred = final_svm_model.predict(X_val_tfidf)

train_accuracy = accuracy_score(y_train,y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Best C: {best_c}")
print(f"Total Support Vectors: {total_support_vectors}")
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Define the SVM model with the best hyperparameter (C=1)
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X_train)
best_svm_model = SVC(kernel='linear', C=2)
best_svm_model.fit(X_tfidf, y_train)

#Get weights and features name
feature_names = vectorizer.get_feature_names_out()
coefficients = best_svm_model.coef_.toarray()[0]

coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
sorted_coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

#Find the top ten
top_positive_words = sorted_coef_df.head(10)
top_negative_words = sorted_coef_df.tail(10)

print("Top 10 words with the highest positive coefficients:")
print(top_positive_words)

print("\nTop 10 words with the highest negative coefficients:")
print(top_negative_words)

# Define the SVM model with RBF kernel
svm_rbf_model = SVC(kernel='rbf')

# Define the hyperparameter grid for grid search
param_grid = {'C': [10**i for i in range(-3,4)], 'gamma': [10**(-i) for i in range(-3,2)]}

# Perform grid search with cross-validation based on validation accuracy
grid_search_rbf = GridSearchCV(svm_rbf_model, param_grid, cv=3, scoring='accuracy')
grid_search_rbf.fit(X_train_tfidf, y_train)

# Get the best hyperparameters
best_params_rbf = grid_search_rbf.best_params_
best_c_rbf = best_params_rbf['C']
best_gamma_rbf = best_params_rbf['gamma']

# Train the SVM model with the best hyperparameters on the entire training set
final_svm_rbf_model = SVC(kernel='rbf', C=best_c_rbf, gamma=best_gamma_rbf)
final_svm_rbf_model.fit(X_train_tfidf, y_train)

total_support_vectors = final_svm_rbf_model.n_support_.sum()

y_val_pred_rbf = final_svm_rbf_model.predict(X_val_tfidf)
y_train_pred_rbf = final_svm_rbf_model.predict(X_train_tfidf)

# Evaluate the validation accuracy
val_accuracy_rbf = accuracy_score(y_val, y_val_pred_rbf)
train_accuracy_rbf = accuracy_score(y_train, y_train_pred_rbf)

print(f"Best C for RBF: {best_c_rbf}")
print(f"Best Gamma for RBF: {best_gamma_rbf}")
print(f"Validation Accuracy with RBF Kernel: {val_accuracy_rbf:.4f}")
print(f"Training Accuracy with RBF Kernel: {train_accuracy_rbf:.4f}")

"""1) As C increases, training accuracy is expected to increase, as the model becomes less regularized and fits the training data more closely. However, if C becomes too large, the model may overfit, and training accuracy may reach near-perfect levels. When gamma decreases, training accuracy is expected to increase, as the decision boundary becomes smoother and less sensitive to individual data points. However, if gamma is too small, the model may underfit the training data.

2) Similarly, with a low value of C, the model is expected to have higher bias and may underfit the training data, whereas a lower gamma results in a smoother decision boundary. However, if gamma is too low, the model might oversimplify and underfit the training data, leading to reduced validation accuracy. As a result, if validation accuracy is significantly lower than training accuracy, it might indicate overfitting. If both accuracies are low, it could suggest underfitting.

3) Lower C and gamma lead to larger margins and simpler decision boundaries, potentially leading to fewer support vectors, whereas higher hyperparameters may result in a larger number of support vectors.
"""

# Your code go# Define the SVM model with RBF kernel
# Grid Refinement for Central Optima (1)

svm_rbf_model = SVC(kernel='rbf')

# Define the hyperparameter grid for grid search. We will do grid refindment around 10 and .1:
param_grid = {'C': [5,10,20,40,70], 'gamma': [.05,.1,.2,.5,.75]}

# Perform grid search with cross-validation based on validation accuracy
grid_search_rbf = GridSearchCV(svm_rbf_model, param_grid, cv=3, scoring='accuracy')
grid_search_rbf.fit(X_train_tfidf, y_train)

# Get the best hyperparameters
best_params_rbf = grid_search_rbf.best_params_
best_c_rbf = best_params_rbf['C']
best_gamma_rbf = best_params_rbf['gamma']

# Train the SVM model with the best hyperparameters on the entire training set
final_svm_rbf_model = SVC(kernel='rbf', C=best_c_rbf, gamma=best_gamma_rbf)
final_svm_rbf_model.fit(X_train_tfidf, y_train)

"""# Naive Bayes Classifier

"""

alpha_values = [2, 1, 0.5, 0.1, 0.05, 0.01]

train_accuracies = []
val_accuracies = []

for alpha in alpha_values:
    # Use TfidfVectorizer to convert text data to TF-IDF features
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    # Build a Multinomial Naive Bayes model with the specified alpha
    nb_model = MultinomialNB(alpha=alpha)

    # Train the model on the training set
    nb_model.fit(X_train_tfidf, y_train)

    # Make predictions on the validation set
    y_val_pred = nb_model.predict(X_val_tfidf)
    y_train_pred = nb_model.predict(X_train_tfidf)

    # Evaluate the accuracy on the validation set
    accuracy_val = accuracy_score(y_val, y_val_pred)
    accuracy_train = accuracy_score(y_train, y_train_pred)

    print(f"Validation Accuracy with Multinomial Naive Bayes (alpha={alpha}): {accuracy_val:.4f}")
    print(f"Training Accuracy with Multinomial Naive Bayes (alpha={alpha}): {accuracy_train:.4f}")
    val_accuracies.append(accuracy_val)
    train_accuracies.append(accuracy_train)

plt.figure(figsize=(10, 6))
plt.semilogx(alpha_values, train_accuracies, label='Training Accuracy', marker='o')
plt.semilogx(alpha_values, val_accuracies, label='Validation Accuracy', marker='o')

plt.title('Accuracy vs. Alpha')
plt.xlabel('Alpha (log scale)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

"""If alpha is very low, the model becomes less robust to unseen features. It may assign zero probability to features not present in the training data, leading to overfitting. As a result, training accuracy may be high because the model is fitting the training data closely, but validation accuracy is likely to be low due to poor generalization. If alpha is high, the model is more robust to unseen features, as it assigns non-zero probabilities to all features. Training accuracy may be slightly lower compared to the case of low alpha because the model is less likely to fit the noise in the training data. However, validation accuracy is expected to be higher - which is not exactly what we see in our plot (can be caused by the quality of dataset we are working with or we may need to check for larger alpha since the difference is slowly becoming smaller)."""

alpha_values = [10 , 5, 2, 1.5, 1, 0.5, 0.1, 0.07, 0.05, 0.03, 0.01,.001,.0001]

for alpha in alpha_values:
    # Use TfidfVectorizer to convert text data to TF-IDF features
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    # Build a Multinomial Naive Bayes model with the specified alpha
    nb_model = MultinomialNB(alpha=alpha)

    # Train the model on the training set
    nb_model.fit(X_train_tfidf, y_train)

    # Make predictions on the validation set
    y_val_pred = nb_model.predict(X_val_tfidf)
    y_train_pred = nb_model.predict(X_train_tfidf)

    # Evaluate the accuracy on the validation set
    accuracy_val = accuracy_score(y_val, y_val_pred)
    accuracy_train = accuracy_score(y_train, y_train_pred)

    print(f"Validation Accuracy with Multinomial Naive Bayes (alpha={alpha}): {accuracy_val:.4f}")

"""The best validation accuracy comes when alpha = .03 in this case.


"""

alpha_1 = 1
alpha_0_03 = 0.03

nb_model_1 = MultinomialNB(alpha=alpha_1)
nb_model_0_03 = MultinomialNB(alpha=alpha_0_03)
nb_model_1.fit(X_tfidf, y_train)
nb_model_0_03.fit(X_tfidf, y_train)

feature_names = vectorizer.get_feature_names_out()

# Get the log probabilities for each feature in both models
log_probs_1 = nb_model_1.feature_log_prob_
log_probs_0_03 = nb_model_0_03.feature_log_prob_


sorted_indices_1 = log_probs_1.argsort(axis=1)[:, ::-1]
sorted_indices_0_03 = log_probs_0_03.argsort(axis=1)[:, ::-1]

# Print the top 10 words for each model
print("Top 10 words with the highest positive weights and their corresponding weights (alpha=1):")
for i in range(10):
    print(f"{feature_names[sorted_indices_1[1, i]]}: {log_probs_1[1, sorted_indices_1[1, i]]:.4f}")

print("\nTop 10 words with the highest negative weights and their corresponding weights (alpha=1):")
for i in range(10):
    print(f"{feature_names[sorted_indices_1[0, i]]}: {log_probs_1[0, sorted_indices_1[0, i]]:.4f}")

print("\nTop 10 words with the highest positive weights and their corresponding weights (alpha=0.03):")
for i in range(10):
    print(f"{feature_names[sorted_indices_0_03[1, i]]}: {log_probs_0_03[1, sorted_indices_0_03[1, i]]:.4f}")

print("\nTop 10 words with the highest negative weights and their corresponding weights (alpha=0.03):")
for i in range(10):
    print(f"{feature_names[sorted_indices_0_03[0, i]]}: {log_probs_0_03[0, sorted_indices_0_03[0, i]]:.4f}")

"""1) The list of words (both positive and negative) are identical, but the weights are lower (if we talking about magnitude) when the alpha is larger. Since the NB classifier is less robust to the outliers when alpha is larger, it tends to increase the magnitude of the highest weights.

2) For the best linear SVM model, the top 10 negative and positive words make more sense since they are nouns, verbs and adjectives vs. mostly pronouns and prepositions in the NB model. It seems that the NB model tends to overfit the data more, assigning larger weights to words that do not fully convey the meaning of being "positive" or "negative", hence this model does not perform very well comared to the linear SVM.
"""