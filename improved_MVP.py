# 17.6.25
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader
import re

# Paths to CSV files
REVIEWS_PATH = "C:/Users/TLP-001/.cache/kagglehub/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews/versions/4/rotten_tomatoes_movie_reviews.csv"
MOVIES_PATH = "C:/Users/TLP-001/.cache/kagglehub/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews/versions/4/rotten_tomatoes_movies.csv"
REVIEWERS_NUM = 1000  # Number of top critics to keep
EPOCHS = 6
MODEL_NAME = "models/torch_model"

# Helper function
def split_features(field):
    if pd.isna(field):
        return []
    return [f.strip() for f in re.split(r'[|,]', field) if f.strip()]

# Dataset class
class MovieReviewDataset(Dataset):
    def __init__(self, users, items, labels, item_side_features):
        self.users = users
        self.items = items
        self.labels = labels
        self.item_side_features = item_side_features

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.users[idx],
            self.items[idx],
            self.item_side_features[self.items[idx]],
            self.labels[idx]
        )

# Recommender model
class HybridRecommender(nn.Module):
    def __init__(self, num_users, num_items, num_side_features, embedding_dim=64):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embedding_dim)
        self.item_embed = nn.Embedding(num_items, embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2 + num_side_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, user_ids, item_ids, item_features):
        u = self.user_embed(user_ids)
        i = self.item_embed(item_ids)
        x = torch.cat([u, i, item_features], dim=1)
        return self.fc(x).squeeze()

# Prepare data
def prepare_data():
    df_reviews = pd.read_csv(REVIEWS_PATH)
    df_movies = pd.read_csv(MOVIES_PATH)

    top_critics = df_reviews['criticName'].value_counts().head(REVIEWERS_NUM).index
    df_reviews = df_reviews[df_reviews['criticName'].isin(top_critics)]
    df_reviews.dropna(subset=['reviewText', 'scoreSentiment'], inplace=True)
    df_reviews['label'] = df_reviews['scoreSentiment'].map({'POSITIVE': 1, 'NEGATIVE': 0})
    df_reviews['id'] = df_reviews['id'].str.strip().str.lower()
    df_movies['title'] = df_movies['title'].str.strip().str.lower()

    common_movies = set(df_reviews['id']).intersection(df_movies['title'])
    df_reviews = df_reviews[df_reviews['id'].isin(common_movies)]
    df_movies = df_movies[df_movies['title'].isin(common_movies)]

    user2idx = {u: i for i, u in enumerate(df_reviews['criticName'].unique())}
    item2idx = {m: i for i, m in enumerate(df_movies['title'].unique())}

    genre_list, lang_list, dir_list = [], [], []
    for _, row in df_movies.iterrows():
        genre_list.append(split_features(row['genre']))
        lang_list.append(split_features(row['originalLanguage']))
        dir_list.append(split_features(row['director']))

    mlb_genre, mlb_lang, mlb_dir = MultiLabelBinarizer(), MultiLabelBinarizer(), MultiLabelBinarizer()
    genre_feat = mlb_genre.fit_transform(genre_list)
    lang_feat = mlb_lang.fit_transform(lang_list)
    dir_feat = mlb_dir.fit_transform(dir_list)

    item_features = np.hstack([genre_feat, lang_feat, dir_feat])
    item_features_tensor = torch.tensor(item_features, dtype=torch.float32)

    users = df_reviews['criticName'].map(user2idx).values
    items = df_reviews['id'].map(item2idx).values
    labels = df_reviews['label'].values

    return train_test_split(users, items, labels, test_size=0.2, random_state=42), item_features_tensor, len(user2idx), len(item2idx), item2idx

# Train model
def train_model():
    (train_users, test_users, train_items, test_items, train_labels, test_labels), item_features_tensor, num_users, num_items, item2idx = prepare_data()

    train_dataset = MovieReviewDataset(
        torch.tensor(train_users, dtype=torch.long),
        torch.tensor(train_items, dtype=torch.long),
        torch.tensor(train_labels, dtype=torch.float32),
        item_features_tensor
    )
    print("Label distribution:", np.bincount(train_labels))
    print("Number of users:", num_users, "Number of items:", num_items)
    test_dataset = MovieReviewDataset(
        torch.tensor(test_users, dtype=torch.long),
        torch.tensor(test_items, dtype=torch.long),
        torch.tensor(test_labels, dtype=torch.float32),
        item_features_tensor
    )

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    model = HybridRecommender(num_users, num_items, item_features_tensor.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for user_ids, item_ids, item_feats, y in train_loader:
            optimizer.zero_grad()
            y_pred = model(user_ids, item_ids, item_feats)
            loss = loss_fn(y_pred, y)
            loss.backward()
            # optimizer = optim.Adam(model.parameters(), lr=1e-3)
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for user_ids, item_ids, item_feats, y in test_loader:
            y_pred = model(user_ids, item_ids, item_feats)
            all_preds.extend(y_pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    preds_bin = [1 if p > 0.5 else 0 for p in all_preds]
    acc = accuracy_score(all_labels, preds_bin)
    auc = roc_auc_score(all_labels, all_preds)
    print(f"Test Accuracy: {acc:.4f}, AUC: {auc:.4f}")

    torch.save(model.state_dict(), f"{MODEL_NAME}_epochs_{EPOCHS}_accuracy_{acc:.4f}_auc_{auc:.4f}.pt")
    return model, item_features_tensor, item2idx

# Recommend for existing user
def recommend(model, user_idx, item_features_tensor, item2idx, top_n=10):
    model.eval()
    item_ids = torch.arange(len(item2idx))
    user_ids = torch.full_like(item_ids, user_idx)
    with torch.no_grad():
        scores = model(user_ids, item_ids, item_features_tensor)
    top_items = torch.topk(scores, top_n).indices.numpy()
    idx2item = {v: k for k, v in item2idx.items()}
    return [idx2item[i] for i in top_items]

# Recommend for a new movie (cold start)
def recommend_new_movie(model, user_idx, new_movie_features_tensor):
    model.eval()
    with torch.no_grad():
        u = model.user_embed(torch.tensor([user_idx]))
        dummy_item = model.item_embed(torch.tensor([0])) * 0
        x = torch.cat([u, dummy_item, new_movie_features_tensor.unsqueeze(0)], dim=1)
        score = model.fc(x).item()
    return score

# Entry point
if __name__ == "__main__":
    model, item_features_tensor, item2idx = train_model()
    # Example: Get recommendations
    # recs = recommend(model, user_idx=0, item_features_tensor=item_features_tensor, item2idx=item2idx)
    # print("Recommendations:", recs)

    # Example: Predict for a new movie (cold start)
    # new_movie_features = torch.zeros(item_features_tensor.shape[1])  # Replace with actual feature vector
    # score = recommend_new_movie(model, user_idx=0, new_movie_features_tensor=new_movie_features)
    # print("Score for new movie:", score)
