
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
df = pd.read_csv("app_user_behavior_dataset.csv")


print("FIRST 5 ROWS:")
print(df.head())

# Check number of rows and columns
print("\nDATASET SHAPE:")
print(df.shape)

# Check column names and data types
print("\nDATASET INFORMATION:")
print(df.info())

# Check missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())


print("\nOVERALL APP USAGE ANALYSIS")

print("Total Users:", df["user_id"].nunique())

print("Average Daily Usage (minutes):",
      round(df["daily_active_minutes"].mean(), 2))

print("Maximum Daily Usage (minutes):",
      df["daily_active_minutes"].max())

print("Minimum Daily Usage (minutes):",
      df["daily_active_minutes"].min())

print("Average Weekly Sessions:",
      round(df["sessions_per_week"].mean(), 2))

print("Average Session Duration (minutes):",
      round(df["avg_session_duration_min"].mean(), 2))


print("\nAGE ANALYSIS")

print("Minimum Age:", df["age"].min())
print("Maximum Age:", df["age"].max())

print("\nAge Distribution:")
print(df["age"].describe())


# Create age groups
bins = [17, 25, 40, 59]
labels = ["18-25", "26-40", "41-59"]

df["age_group"] = pd.cut(
    df["age"],
    bins=bins,
    labels=labels
)

# Age-wise analysis
age_analysis = df.groupby(
    "age_group", observed=False
).agg(
    total_users=("user_id", "nunique"),
    average_daily_usage=("daily_active_minutes", "mean"),
    average_weekly_sessions=("sessions_per_week", "mean")
).round(2)

print("\nAGE-WISE APP USAGE ANALYSIS")
print(age_analysis)


# STEP 1: DUPLICATE CHECKING

print("\nDUPLICATE RECORD ANALYSIS")

print("Total Duplicate Rows:",
      df.duplicated().sum())

print("Duplicate User IDs:",
      df["user_id"].duplicated().sum())


# STEP 2: INVALID VALUE CHECKING

print("\nINVALID VALUE ANALYSIS")

print("Invalid Age:",
      (df["age"] <= 0).sum())

print("Invalid Daily Usage:",
      (df["daily_active_minutes"] < 0).sum())

print("Invalid Session Duration:",
      (df["avg_session_duration_min"] < 0).sum())

print("Invalid Weekly Sessions:",
      (df["sessions_per_week"] < 0).sum())


# STEP 3: MISSING VALUE PERCENTAGE

print("\nMISSING VALUE PERCENTAGE")

missing_count = df["rating_given"].isnull().sum()

missing_percentage = (
    missing_count / len(df)
) * 100

print("Missing Rating Count:", missing_count)

print("Missing Rating Percentage:",
      round(missing_percentage, 2), "%")


# SAVE WORKING DATASET

df.to_csv("app_user_behavior_working.csv", index=False)

print("\nWorking dataset saved successfully!")


# STEP 5: NUMERICAL DATA SUMMARY

print("\nNUMERICAL DATA SUMMARY")

print(df.describe())


# STEP 6: CHECK NUMERICAL COLUMN RANGES

print("\nNUMERICAL COLUMN MINIMUM VALUES")
print(df.select_dtypes(include="number").min())

print("\nNUMERICAL COLUMN MAXIMUM VALUES")
print(df.select_dtypes(include="number").max())

# STEP 7: CHECK CATEGORICAL COLUMN VALUES

categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    print("\n", column)
    print(df[column].value_counts())

# STEP 8: FEATURE SELECTION

behavior_features = [
    "sessions_per_week",
    "avg_session_duration_min",
    "daily_active_minutes",
    "feature_clicks_per_session",
    "notifications_opened_per_week",
    "in_app_search_count",
    "pages_viewed_per_session",
    "crash_events_last_30_days",
    "support_tickets_raised",
    "days_since_last_login",
    "ads_clicked_last_30_days",
    "content_downloads",
    "social_shares",
    "account_age_days"
]

X = df[behavior_features]

print("\nSELECTED BEHAVIOUR FEATURES")
print(X.head())

print("\nFEATURE DATASET SHAPE")
print(X.shape)

print("\nMISSING VALUES IN SELECTED FEATURES")
print(X.isnull().sum())
from sklearn.preprocessing import StandardScaler

# STEP 9: FEATURE SCALING

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nSCALED DATASET SHAPE")
print(X_scaled.shape)

print("\nFIRST 5 ROWS OF SCALED DATA")
print(X_scaled[:5])

print("\nSCALED DATA MEAN")
print(X_scaled.mean(axis=0).round(2))

print("\nSCALED DATA STANDARD DEVIATION")
print(X_scaled.std(axis=0).round(2))

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# STEP 10: ELBOW METHOD AND SILHOUETTE SCORE

wcss = []

print("\nWCSS VALUES")

for i in range(2, 9):

    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

print(wcss)


# SILHOUETTE SCORE

print("\nSILHOUETTE SCORES")

for i in range(2, 9):

    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        cluster_labels,
        sample_size=5000,
        random_state=42
    )

    print("Clusters:", i, "| Score:", round(score, 4))


# DISPLAY ELBOW GRAPH

plt.figure(figsize=(8, 5))

plt.plot(range(2, 9), wcss, marker="o")

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.xticks(range(2, 9))
plt.grid(True)

plt.show()

# STEP 11: FINAL KMEANS CLUSTERING

print("\nFINAL KMEANS MODEL")

# Create KMeans model with 2 clusters
kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

# Assign cluster labels to users
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Count users in each cluster
print("\nUSERS IN EACH CLUSTER")

print(df["Cluster"].value_counts().sort_index())

# Calculate percentage of users
print("\nCLUSTER PERCENTAGE")

print(
    (df["Cluster"].value_counts(normalize=True)
     .sort_index() * 100).round(2)
)

# Save clustered dataset
df.to_csv("app_user_behavior_clustered.csv", index=False)

print("\nClustered dataset saved successfully!")


# STEP 12: CLUSTER ANALYSIS

print("\nCLUSTER-WISE USER BEHAVIOUR ANALYSIS")

cluster_analysis = df.groupby("Cluster")[behavior_features].mean().round(2)

print("\nCOMPLETE CLUSTER-WISE BEHAVIOUR ANALYSIS")

print(cluster_analysis.T)

# Save cluster analysis
cluster_analysis.to_csv("cluster_analysis.csv")

print("\nCluster analysis saved successfully!")



# STEP 13: SUPPORT TICKET ANALYSIS BY CLUSTER

print("\nSUPPORT TICKET DISTRIBUTION BY CLUSTER")

print(
    pd.crosstab(
        df["Cluster"],
        df["support_tickets_raised"]
    )
)

print("\nSUPPORT TICKET MINIMUM AND MAXIMUM")

print(
    df.groupby("Cluster")["support_tickets_raised"]
    .agg(["min", "max", "mean"])
)

# STEP 14: IMPROVED FEATURE SELECTION

behavior_features = [
    "sessions_per_week",
    "avg_session_duration_min",
    "daily_active_minutes",
    "feature_clicks_per_session",
    "notifications_opened_per_week",
    "in_app_search_count",
    "pages_viewed_per_session",
    "crash_events_last_30_days",
    "days_since_last_login",
    "ads_clicked_last_30_days",
    "content_downloads",
    "social_shares",
    "account_age_days"
]

X = df[behavior_features]

print("\nIMPROVED FEATURE DATASET SHAPE")
print(X.shape)

print("\nMISSING VALUES")
print(X.isnull().sum())

# STEP 15: FEATURE SCALING

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nIMPROVED SCALED DATA SHAPE")
print(X_scaled.shape)

print("\nSCALED DATA MEAN")
print(X_scaled.mean(axis=0).round(2))

print("\nSCALED DATA STANDARD DEVIATION")
print(X_scaled.std(axis=0).round(2))


# STEP 16: IMPROVED ELBOW METHOD

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wcss = []

for i in range(2, 9):

    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)

print("\nIMPROVED WCSS VALUES")
print(wcss)

plt.figure(figsize=(8, 5))

plt.plot(range(2, 9), wcss, marker='o')

plt.title("Improved Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

plt.show()

# STEP 17: IMPROVED SILHOUETTE SCORE

from sklearn.metrics import silhouette_score

print("\nIMPROVED SILHOUETTE SCORES")

for i in range(2, 9):

    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels,
        sample_size=5000,
        random_state=42
    )

    print(f"Clusters: {i} | Score: {score:.4f}")

    # STEP 18: IMPROVED FINAL KMEANS MODEL

kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

df["Improved_Cluster"] = kmeans.fit_predict(X_scaled)

print("\nIMPROVED USERS IN EACH CLUSTER")
print(df["Improved_Cluster"].value_counts().sort_index())

print("\nIMPROVED CLUSTER PERCENTAGE")
print(
    (df["Improved_Cluster"].value_counts(normalize=True).sort_index() * 100)
    .round(2)
)

df.to_csv("app_user_behavior_improved_clustered.csv", index=False)

print("\nIMPROVED CLUSTERED DATASET SAVED SUCCESSFULLY!")

# STEP 19: IMPROVED CLUSTER-WISE USER BEHAVIOUR ANALYSIS

print("\nIMPROVED CLUSTER-WISE USER BEHAVIOUR")

cluster_analysis = df.groupby("Improved_Cluster")[behavior_features].mean().round(2)

print(cluster_analysis.to_string())

# STEP 20: ACCOUNT AGE ANALYSIS BY CLUSTER

print("\nACCOUNT AGE MINIMUM AND MAXIMUM")

print(
    df.groupby("Improved_Cluster")["account_age_days"]
    .agg(["min", "max", "mean"])
    .round(2)
)

# STEP 21: ENGAGEMENT SCORE ANALYSIS BY CLUSTER

print("\nENGAGEMENT SCORE BY CLUSTER")

print(
    df.groupby("Improved_Cluster")["engagement_score"]
    .agg(["min", "max", "mean"])
    .round(2)
)


# STEP 22: FINAL EXPERIMENT - REMOVE ACCOUNT AGE

behavior_features = [
    "sessions_per_week",
    "avg_session_duration_min",
    "daily_active_minutes",
    "feature_clicks_per_session",
    "notifications_opened_per_week",
    "in_app_search_count",
    "pages_viewed_per_session",
    "crash_events_last_30_days",
    "days_since_last_login",
    "ads_clicked_last_30_days",
    "content_downloads",
    "social_shares"
]

X = df[behavior_features]

print("\nFINAL FEATURE SHAPE:")
print(X.shape)

# STEP 23: SCALE THE FEATURES

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nSCALED DATA SHAPE:")
print(X_scaled.shape)

# STEP 24: CHECK SILHOUETTE SCORES

for k in range(2, 9):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    print(f"K = {k}, Silhouette Score = {score:.4f}")

# STEP 25: FINAL K-MEANS MODEL

final_kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

df["Final_Cluster"] = final_kmeans.fit_predict(X_scaled)

print("\nFINAL CLUSTER DISTRIBUTION:")
print(df["Final_Cluster"].value_counts().sort_index())

# Save final dataset
df.to_csv("app_user_behavior_final_clustered.csv", index=False)

print("\nFinal clustered dataset saved successfully!")


# STEP 26: FINAL CLUSTER BEHAVIOUR ANALYSIS

print("\nFINAL CLUSTER AVERAGES")

print(
    df.groupby("Final_Cluster")[behavior_features]
    .mean()
    .round(2)
)

print("\nENGAGEMENT SCORE BY FINAL CLUSTER")

print(
    df.groupby("Final_Cluster")["engagement_score"]
    .agg(["min", "max", "mean"])
    .round(2)
)



# ==========================================
# PROFESSIONAL COLOURFUL PROJECT VISUALIZATION
# STEP 27 TO STEP 31
# ==========================================

import matplotlib.pyplot as plt
import numpy as np

# Professional chart styling
plt.rcParams.update({
    "figure.facecolor": "#F7F8FC",
    "axes.facecolor": "white",
    "axes.titleweight": "bold",
    "axes.titlesize": 16,
    "axes.labelsize": 11,
    "font.size": 10,
    "grid.color": "#E5E7EB",
    "grid.linestyle": "--",
    "grid.alpha": 0.7
})

cluster_colors = ["#6554C0", "#16A6A1"]


# STEP 27: FINAL CLUSTER DISTRIBUTION CHART

print("\nFINAL CLUSTER DISTRIBUTION CHART")

cluster_counts = df["Final_Cluster"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(9, 6))

bars = ax.bar(
    ["Cluster 0", "Cluster 1"],
    cluster_counts.values,
    color=cluster_colors,
    width=0.55
)

ax.set_title("App User Distribution by Cluster", pad=18)
ax.set_xlabel("User Clusters")
ax.set_ylabel("Number of Users")
ax.grid(axis="y")
ax.set_axisbelow(True)

for bar, value in zip(bars, cluster_counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 200,
        f"{value:,}\n({value / len(df) * 100:.2f}%)",
        ha="center",
        fontweight="bold",
        fontsize=11
    )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("final_cluster_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

print("Cluster distribution chart saved!")


# STEP 28: PROFESSIONAL CLUSTER BEHAVIOUR HEATMAP

print("\nCLUSTER BEHAVIOUR COMPARISON")

scaled_df = pd.DataFrame(
    X_scaled,
    columns=behavior_features
)

scaled_df["Final_Cluster"] = df["Final_Cluster"].values

cluster_profile = scaled_df.groupby("Final_Cluster").mean()

# Short and clear feature names
short_names = [
    "Sessions / Week",
    "Session Duration",
    "Daily Usage",
    "Feature Clicks",
    "Notifications",
    "App Searches",
    "Pages Viewed",
    "Crash Events",
    "Days Since Login",
    "Ad Clicks",
    "Downloads",
    "Social Shares"
]

# Transpose for horizontal layout
heatmap_data = cluster_profile.T

fig, ax = plt.subplots(figsize=(9, 9))

im = ax.imshow(
    heatmap_data.values,
    aspect="auto",
    cmap="viridis"
)

# Feature names on left
ax.set_yticks(range(len(short_names)))
ax.set_yticklabels(short_names, fontsize=10)

# Cluster names on top
ax.set_xticks([0, 1])
ax.set_xticklabels(
    ["Cluster 0", "Cluster 1"],
    fontsize=11,
    fontweight="bold"
)

ax.xaxis.tick_top()

ax.set_title(
    "User Behaviour Comparison",
    fontsize=17,
    fontweight="bold",
    pad=25
)

# Display values inside cells
for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):

        value = heatmap_data.iloc[i, j]

        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold"
        )

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Average Scaled Value")

ax.set_xticks(
    np.arange(-0.5, 2, 1),
    minor=True
)

ax.set_yticks(
    np.arange(-0.5, len(short_names), 1),
    minor=True
)

ax.grid(which="minor", color="white", linewidth=2)
ax.tick_params(which="minor", bottom=False, left=False)

plt.tight_layout()

plt.savefig(
    "cluster_behavior_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Professional heatmap saved successfully!")


# STEP 29: DISPLAY FEATURE AVERAGES

print("\nSCALED FEATURE AVERAGES BY CLUSTER:")
print(cluster_profile.T.round(3).to_string())


# STEP 30: ACTUAL PAGES VIEWED BY CLUSTER

print("\nPAGES VIEWED PER SESSION BY CLUSTER")

pages_analysis = df.groupby("Final_Cluster")[
    "pages_viewed_per_session"
].agg(["count", "min", "max", "mean", "median"]).round(2)

print(pages_analysis)


# STEP 31: ENGAGEMENT SCORE COMPARISON

print("\nENGAGEMENT SCORE COMPARISON")

engagement_means = (
    df.groupby("Final_Cluster")["engagement_score"].mean()
)

fig, ax = plt.subplots(figsize=(9, 6))

bars = ax.bar(
    ["Cluster 0", "Cluster 1"],
    engagement_means.values,
    color=cluster_colors,
    width=0.55
)

ax.set_title("Average Engagement Score by Cluster", pad=18)
ax.set_xlabel("User Clusters")
ax.set_ylabel("Average Engagement Score")
ax.set_ylim(0, 100)
ax.grid(axis="y")
ax.set_axisbelow(True)

for bar, value in zip(bars, engagement_means.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}",
        ha="center",
        fontweight="bold",
        fontsize=12
    )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("engagement_score_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

print("Engagement comparison chart saved!")

print("\nALL FINAL PROJECT CHARTS GENERATED SUCCESSFULLY!")
