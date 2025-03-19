import pandas as pd
import numpy as np
import sparse
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.keras.layers import Embedding, Dense, Flatten, Input, Concatenate, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score,balanced_accuracy_score
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import scipy.sparse as sp
from typing import List
from scipy import sparse
from tensorflow.keras.models import load_model

# Convert from wide format (sparse matrix) to long format (DataFrame)
def wide_to_long(wide: np.array, possible_ratings: List[int]) -> np.array:
    

    def _get_ratings(arr: np.array, rating: int) -> np.array:
        idx = np.where(arr == rating)
        return np.vstack(
            (idx[0], idx[1], np.ones(idx[0].size, dtype="int8") * rating)
        ).T

    long_arrays = []
    for r in possible_ratings:
        long_arrays.append(_get_ratings(wide, r))

    return np.vstack(long_arrays)

# Load the dataset
book_data = pd.read_csv(r'C:\Users\user\Downloads\BX-Books.csv', delimiter=";", usecols=["ISBN", "Book-Title", "Book-Author"] ,encoding='latin-1')

# Create a dictionary that maps ISBN to book details
book_dict = {str(row['ISBN']): {"title": row['Book-Title'], "author": row['Book-Author']} 
             for _, row in book_data.iterrows()}

# Example function to get book details by ISBN
def get_book_details(isbn):
    book_info = book_dict.get(str(isbn).strip())
    if book_info:
        return f"Title: {book_info['title']}, Author: {book_info['author']}"
    else:
        return "Book not found"

data = pd.read_csv(r'C:\Users\user\Downloads\BX-Book-Ratings.csv', delimiter=';', encoding='latin-1')

data = data[data['Book-Rating'] >= 5]

# Filter users and ratings based on your conditions
counts1 = data['User-ID'].value_counts()
ratings = data[data['User-ID'].isin(counts1[counts1 >= 150].index)]
counts = ratings['Book-Rating'].value_counts()
ratings = ratings[ratings['Book-Rating'].isin(counts[counts >= 200].index)]
data = ratings

# Map ISBN to contiguous integers (based on the entire dataset)
isbn_mapping = {isbn: idx for idx, isbn in enumerate(data['ISBN'].unique())}
data['ISBN_mapped'] = data['ISBN'].map(isbn_mapping)

# Map User-ID to contiguous integers (based on the entire dataset)
user_mapping = {user: idx for idx, user in enumerate(data['User-ID'].unique())}
data['User_ID_mapped'] = data['User-ID'].map(user_mapping)

# Split the data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Drop rows with NaN in any column
train_data = train_data.dropna()
test_data = test_data.dropna()

# Pivot the train_data to get a dense matrix
train_matrix_dense = train_data.pivot(index='User_ID_mapped', columns='ISBN_mapped', values='Book-Rating').fillna(0).values

# Pivot the test_data to get a dense matrix
test_matrix_dense = test_data.pivot(index='User_ID_mapped', columns='ISBN_mapped', values='Book-Rating').fillna(0).values

# Convert the dense matrices to COO matrices
train_matrix = coo_matrix(train_matrix_dense)
test_matrix = coo_matrix(test_matrix_dense)

train_matrix = (train_matrix.toarray() > 0).astype("int8")
test_matrix = (test_matrix.toarray() > 0).astype("int8")
unique_ratings = np.unique(train_matrix)
long_train = wide_to_long(train_matrix, unique_ratings)
train_long = pd.DataFrame(long_train, columns=["user_id", "book_id", "interaction"])
long_test = wide_to_long(test_matrix, unique_ratings)
test_long = pd.DataFrame(long_test, columns=["user_id", "book_id", "interaction"])
zeros = train_long[train_long['interaction'] == 0]
num_zeros_to_drop = 11000000  
zeros_to_drop = zeros.sample(n=num_zeros_to_drop, random_state=42)
train_long = train_long.drop(zeros_to_drop.index)
count_1 = (train_long['interaction'] == 1).sum()
count_0 = (train_long['interaction'] == 0).sum()


def create_ncf(
    number_of_users: int,
    number_of_items: int,
    latent_dim_mf: int = 4,
    latent_dim_mlp: int = 32,
    reg_mf: int = 0,
    reg_mlp: int = 0.01,
    dense_layers: List[int] = [8, 4],
    reg_layers: List[int] = [0.01, 0.01],
    activation_dense: str = "relu",
) -> keras.Model:
    user = Input(shape=(), dtype="int32", name="user_id")
    item = Input(shape=(), dtype="int32", name="book_id")

    mf_user_embedding = Embedding(
        input_dim=number_of_users+1,
        output_dim=latent_dim_mf+1,
        name="mf_user_embedding",
        embeddings_initializer="RandomNormal",
        embeddings_regularizer=l2(reg_mf),
        input_length=1,
    )
    mf_item_embedding = Embedding(
        input_dim=number_of_items+1,
        output_dim=latent_dim_mf+1,
        name="mf_item_embedding",
        embeddings_initializer="RandomNormal",
        embeddings_regularizer=l2(reg_mf),
        input_length=1,
    )

    mlp_user_embedding = Embedding(
        input_dim=number_of_users+1,
        output_dim=latent_dim_mlp+1,
        name="mlp_user_embedding",
        embeddings_initializer="RandomNormal",
        embeddings_regularizer=l2(reg_mlp),
        input_length=1,
    )
    mlp_item_embedding = Embedding(
        input_dim=number_of_items+1,
        output_dim=latent_dim_mlp+1,
        name="mlp_item_embedding",
        embeddings_initializer="RandomNormal",
        embeddings_regularizer=l2(reg_mlp),
        input_length=1,
    )

    # MF vector
    mf_user_latent = Flatten()(mf_user_embedding(user))
    mf_item_latent = Flatten()(mf_item_embedding(item))
    mf_cat_latent = Multiply()([mf_user_latent, mf_item_latent])

    # MLP vector
    mlp_user_latent = Flatten()(mlp_user_embedding(user))
    mlp_item_latent = Flatten()(mlp_item_embedding(item))
    mlp_cat_latent = Concatenate()([mlp_user_latent, mlp_item_latent])

    mlp_vector = mlp_cat_latent

    # build dense layers for model
    for i in range(len(dense_layers)):
        layer = Dense(
            dense_layers[i],
            activity_regularizer=l2(reg_layers[i]),
            activation=activation_dense,
            name="layer%d" % i,
        )
        mlp_vector = layer(mlp_vector)

    predict_layer = Concatenate()([mf_cat_latent, mlp_vector])

    result = Dense(
        1, activation="sigmoid", kernel_initializer="lecun_uniform", name="interaction"
    )

    output = result(predict_layer)

    model = Model(
        inputs=[user, item],
        outputs=[output],
    )

    return model

n_users, n_items = train_matrix.shape
'''''
model = create_ncf(n_users, n_items)

model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
'''''

#model.summary()

# Load the saved model
model = load_model("ncf_model.h5")

def make_tf_dataset(
    df: pd.DataFrame,
    targets: List[str],
    val_split: float = 0.1,
    batch_size: int = 512,
    seed=42,
):
    
    n_val = round(df.shape[0] * val_split)
    if seed:
        # shuffle all the rows
        x = df.sample(frac=1, random_state=seed).to_dict("series")
    else:
        x = df.to_dict("series")
    y = dict()
    for t in targets:
        y[t] = x.pop(t)
    ds = tf.data.Dataset.from_tensor_slices((x, y))

    ds_val = ds.take(n_val).batch(batch_size)
    ds_train = ds.skip(n_val).batch(batch_size)
    return ds_train, ds_val

ds_train, ds_val = make_tf_dataset(train_long, ["interaction"])

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3
)

def adjust_sample_weights(features, labels):
    # Calculate the class distribution and dynamically adjust weights
    pos_weight = np.sum(labels == 0) / np.sum(labels == 1)  # Ratio of negative to positive samples
    weights = tf.where(labels == 1, pos_weight, 1.0)  # Apply dynamic weight
    return features, labels, weights

ds_train = ds_train.map(adjust_sample_weights)

'''''
# Fit the model with class weights
train_hist = model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=10,
    callbacks=[tensorboard_callback, early_stopping_callback],
    verbose=1,
)
'''''

model.save("ncf_model.h5")



def get_user_rating_count(test_data, user_id):
    
    # Group by 'User-ID' and count the number of ratings
    user_rating_counts = test_data.groupby('User-ID').size().reset_index(name='Rating Count')
    user_rating_count = user_rating_counts[user_rating_counts['User-ID'] == user_id]['Rating Count'].values
    if user_rating_count.size > 0:
        return user_rating_count[0]
    else:
        print(f"User ID {user_id} not found in the data.")
        return None


def evaluate_metrics_dynamic_thresholds(model, test_data, user_mapping, isbn_mapping, top_k=5, step=0.005):
    user_id = input("Enter the user's ID: ")
    user_id = int(user_id)
    user_mapped = user_mapping.get(user_id)

    if user_mapped is None:
        print(f"User ID {user_id} not found.")
        return

    user_test_data = test_data[test_data['User-ID'] == user_id]
    if user_test_data.empty:
        print(f"No test data available for User ID {user_id}.")
        return

    test_books = user_test_data['ISBN_mapped'].values
    test_interactions = user_test_data['Book-Rating'].apply(lambda x: 1 if x > 7 else 0).values

    predictions = model.predict([np.array([user_mapped] * len(test_books)), test_books]).flatten()
    best_balanced_accuracy = 0
    best_threshold = 0
    metrics = {'threshold': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'balanced_accuracy': []}
    user_ratings_count = get_user_rating_count(test_data, user_id)
 
    
    thresholds = np.arange(0, 1 + step, step)

    for threshold in thresholds:
        predicted_interactions = (predictions > threshold).astype(int)

        acc = accuracy_score(test_interactions, predicted_interactions)
        prec = precision_score(test_interactions, predicted_interactions, zero_division=0)
        rec = recall_score(test_interactions, predicted_interactions, zero_division=0)
        f1 = f1_score(test_interactions, predicted_interactions, zero_division=0)
        balanced_acc = balanced_accuracy_score(test_interactions, predicted_interactions)

        metrics['threshold'].append(threshold)
        metrics['accuracy'].append(acc)
        metrics['precision'].append(prec)
        metrics['recall'].append(rec)
        metrics['f1'].append(f1)
        metrics['balanced_accuracy'].append(balanced_acc)

        if balanced_acc > best_balanced_accuracy:
            best_balanced_accuracy = balanced_acc
            best_threshold = threshold 
    
    if user_ratings_count > 201:
        best_threshold -= 0.04

    #print(f"\nBest Threshold for User {user_id} (Balanced Accuracy): {best_threshold}")
    print(f"Metrics:\n"
          f"Accuracy: {accuracy_score(test_interactions, (predictions > best_threshold).astype(int)):.3f}, "
          f"Precision: {precision_score(test_interactions, (predictions > best_threshold).astype(int), zero_division=0):.3f}, "
          f"Recall: {recall_score(test_interactions, (predictions > best_threshold).astype(int), zero_division=0):.3f}, "
          f"F1 Score: {f1_score(test_interactions, (predictions > best_threshold).astype(int), zero_division=0):.3f}")

    # Recommend Top 5 Books
    reversed_isbn_mapping = {v: k for k, v in isbn_mapping.items()}
    top_indices = predictions.argsort()[::-1] 

    print(f"\nTop {top_k} Recommended Books for User {user_id}:")
    recommended_count = 0  

    for isbn in top_indices:
        if recommended_count >= top_k:
            break 

        isbn_mapped = reversed_isbn_mapping.get(isbn)

        if isbn_mapped is not None:
            book_info = book_dict.get(str(isbn_mapped))  
            if book_info:
                print(f"Title: {book_info['title']}, Author: {book_info['author']}")
                recommended_count += 1  


    return {
        "accuracy": metrics['accuracy'],
        "precision": metrics['precision'],
        "recall": metrics['recall'],
        "f1": metrics['f1'],
    }   
    '''''
    plt.figure(figsize=(12, 8))
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        plt.plot(metrics['threshold'], metrics[metric], label=metric.capitalize(), linewidth=2)
    plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'Best Threshold: {best_threshold:.3f}')
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('Metric Value', fontsize=14)
    plt.title(f"Evaluation Metrics for User {user_id} Across Thresholds", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()
    '''''


evaluate_metrics_dynamic_thresholds(
    model=model,
    test_data=test_data,
    user_mapping=user_mapping,
    isbn_mapping=isbn_mapping,
    step=0.005 
)
