# -----------------------------------------
# Library Imports
# -----------------------------------------
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from math import pi
from scipy.stats import gaussian_kde
from scipy.stats import wasserstein_distance

# -----------------------------------------
# Data Loading and Initial Preprocessing
# -----------------------------------------

df = pd.read_csv("T1.csv")

# Remove unnecessary columns that are not needed for clustering or analysis
df = df.drop(columns=[
    'Wind Speed (m/s)',
    'Theoretical_Power_Curve (KWh)',
    'Wind Direction (°)'
])

# Convert Active Power column to numeric, forcing invalid values to NaN
df['LV ActivePower (kW)'] = pd.to_numeric(df['LV ActivePower (kW)'], errors='coerce')

# -----------------------------------------
# Zero Value Replacement Using Neighbor Averaging
# -----------------------------------------

window = 2  # Number of neighboring values to consider on each side
values = df['LV ActivePower (kW)'].copy()

# Replace zero values with the mean of nearby non-zero values within a defined window
for i in range(len(values)):
    if values[i] == 0:
        start = max(0, i - window)
        end = min(len(values), i + window + 1)
        neighbors = values[start:end]
        neighbors = neighbors[neighbors != 0]
        if len(neighbors) > 0:
            values[i] = neighbors.mean()
df['LV ActivePower (kW)'] = values

# -----------------------------------------
# Downsampling Data (Every 6th Row)
# -----------------------------------------

df = df.iloc[0::6].reset_index(drop=True)

# df.to_csv("data.csv", index=False)

# -----------------------------------------
# Extracting Date Information and Assigning Seasons
# -----------------------------------------

df['Date/Time'] = pd.to_datetime(df['Date/Time'], format="%d %m %Y %H:%M", errors='coerce')
df['Month'] = df['Date/Time'].dt.month

# Function to map month to seasonal category
def get_season(month):
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    else:
        return 'Winter'

# Assign season to each row
df['Season'] = df['Month'].apply(get_season)
df.to_csv('hybrid.csv', index=False)

# -----------------------------------------
# K-Means Clustering to Classify Load Levels (Low, Medium, High)
# -----------------------------------------

results = []

for season in df['Season'].unique():
    temp = df[df['Season']==season].copy()
    X = temp[['LV ActivePower (kW)']].values

    # Perform KMeans clustering with 3 clusters for load categorization
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_

    # Sort clusters by their centroid values to label them Low, Medium, High
    cluster_centers = kmeans.cluster_centers_.flatten()
    sorted_idx = cluster_centers.argsort()  # ascending order
    label_mapping = {sorted_idx[0]: 'Low', sorted_idx[1]: 'Medium', sorted_idx[2]: 'High'}

    # Assign load label to each data point
    temp['Load_Level'] = [label_mapping[label] for label in labels]

    results.append(temp)

# Merge all seasonal results
final_df = pd.concat(results).reset_index(drop=True)

# Export final dataset containing load level categories
final_df.to_csv("Season_Load_Clustered.csv", index=False)

# -----------------------------------------
# Fast Forward Selection (FFS) for Scenario Reduction
# -----------------------------------------

df = pd.read_csv("Season_Load_Clustered.csv")

# Function implementing Fast Forward Selection for representative scenario extraction
def fast_forward_selection(D, probs, target_N):
    S = D.shape[0]
    U = set(range(S))
    J = []
    min_dist_to_J = np.full(S, np.inf)
    for _ in range(target_N):
        best_i = None
        best_Q = None
        for i in list(U):
            temp_min = np.minimum(min_dist_to_J, D[:, i])
            mask = np.array([j in U and j != i for j in range(S)])
            Qi = np.sum(probs[mask] * temp_min[mask]) if mask.any() else 0.0
            if best_Q is None or Qi < best_Q:
                best_Q = Qi
                best_i = i
        J.append(best_i)
        U.remove(best_i)
        min_dist_to_J = np.minimum(min_dist_to_J, D[:, best_i])
    return np.array(J, dtype=int)

representatives = []

# Apply scenario reduction separately for each season and load level cluster
for season in df['Season'].unique():
    temp_season = df[df['Season'] == season]
    for load_level in ['Low', 'Medium', 'High']:
        group = temp_season[temp_season['Load_Level'] == load_level]
        if not group.empty:
            values = group['LV ActivePower (kW)'].values
            S = len(values)
            # Create 1D distance matrix for scenario distance measurement
            D = np.abs(values[:, None] - values[None, :])
            probs = np.full(S, 1/S)  # uniform probability for all scenarios
            # Select one representative scenario per cluster
            selected_idx = fast_forward_selection(D, probs, target_N=1)[0]
            representatives.append(group.iloc[selected_idx])

# Save selected representatives
rep_df = pd.DataFrame(representatives)
rep_df.to_csv("Season_Load_Representatives.csv", index=False)

# -----------------------------------------
# Statistical Comparison Between Full Dataset and Reduced Representatives
# -----------------------------------------

full_df = pd.read_csv("Season_Load_Clustered.csv")
rep_df = pd.read_csv("Season_Load_Representatives.csv")

seasons = full_df['Season'].unique()
results = []

# Compute mean, median, and variance before and after reduction
for season in seasons:
    full_values = full_df[full_df['Season'] == season]['LV ActivePower (kW)'].values
    reps = rep_df[rep_df['Season'] == season]['LV ActivePower (kW)'].values

    row = {
        'Season': season,
        'Mean_Original': np.mean(full_values),
        'Var_Original': np.var(full_values),
        'Mean_Representatives': np.mean(reps),
        'Var_Representatives': np.var(reps)
    }
    results.append(row)

summary_df = pd.DataFrame(results)

summary_df.to_csv("Season_Representatives_vs_Full_Stats.csv", index=False)

print(summary_df)

# -----------------------------------------
# Metrics Calculation Between Full and Reduced Scenario Sets
# -----------------------------------------

# Define evaluation metrics: Wasserstein, Energy, CRPS, Total Variation Distance
def ws_set(values, reps):
    return wasserstein_distance(values, reps)

def energy_set(values, reps):
    x = np.array(values)
    y = np.array(reps)
    return 2 * np.mean(np.abs(x[:, None] - y)) - np.mean(np.abs(x[:, None] - x)) - np.mean(np.abs(y[:, None] - y))

def crps_set(values, reps):
    values = np.array(values)
    reps = np.array(reps)
    M = len(reps)
    term1 = np.mean(np.abs(values[:, None] - reps[None, :]), axis=1)
    term2 = np.mean(np.abs(reps[:, None] - reps[None, :]))
    crps_values = term1 - 0.5 * term2
    return np.mean(crps_values)

def tvd_set(values, reps, bins=50):
    hist_vals, bin_edges = np.histogram(values, bins=bins, density=True)
    hist_reps, _ = np.histogram(reps, bins=bin_edges, density=True)
    return 0.5 * np.sum(np.abs(hist_vals - hist_reps))

full_df = pd.read_csv("Season_Load_Clustered.csv")
rep_df = pd.read_csv("Season_Load_Representatives.csv")

seasons = full_df['Season'].unique()
results = []

# Compute all metrics for each season
for season in seasons:
    full_values = full_df[full_df['Season'] == season]['LV ActivePower (kW)'].values
    reps = rep_df[rep_df['Season'] == season]['LV ActivePower (kW)'].values

    row = {
        'Season': season,
        'Wasserstein': ws_set(full_values, reps),
        'Energy': energy_set(full_values, reps),
        'CRPS': crps_set(full_values, reps),
        'TVD': tvd_set(full_values, reps)
    }
    results.append(row)

metrics_df = pd.DataFrame(results)
metrics_df.to_csv("Season_Full_vs_Representatives_Metrics.csv", index=False)

print(metrics_df)

# -----------------------------------------
# Visualization of Distributions and Metrics
# -----------------------------------------

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# Load required datasets
full_df = pd.read_csv("Season_Load_Clustered.csv")
rep_df = pd.read_csv("Season_Load_Representatives.csv")
metrics_df = pd.read_csv("Season_Full_vs_Representatives_Metrics.csv")

seasons = full_df['Season'].unique()
colors = {'Full': 'skyblue', 'Reps': 'salmon'}
season_colors = {'Winter':'blue','Spring':'green','Summer':'orange','Autumn':'brown'}

# -----------------------------------------
# Distribution Comparison: Histogram + KDE
# -----------------------------------------

for season in seasons:
    full_values = full_df[full_df['Season'] == season]['LV ActivePower (kW)'].values
    reps_values = rep_df[rep_df['Season'] == season]['LV ActivePower (kW)'].values

    plt.figure(figsize=(8,6))
    plt.hist(full_values, bins=30, alpha=0.6, color=colors['Full'], label='All Scenarios', density=True)
    plt.hist(reps_values, bins=30, alpha=0.6, color=colors['Reps'], label='Representatives', density=True)

    x_grid = np.linspace(min(full_values.min(), reps_values.min()),
                         max(full_values.max(), reps_values.max()), 200)
    kde_full = gaussian_kde(full_values)
    kde_reps = gaussian_kde(reps_values)
    plt.plot(x_grid, kde_full(x_grid), color='blue', linestyle='--')
    plt.plot(x_grid, kde_reps(x_grid), color='red', linestyle='--')

    plt.title(f'{season} Distribution', fontsize=14)
    plt.xlabel('LV ActivePower (kW)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{season}_Distribution.png', dpi=300)
    plt.show()

# -----------------------------------------
# Boxplots for Seasonal Comparison
# -----------------------------------------

for season in seasons:
    full_values = full_df[full_df['Season'] == season]['LV ActivePower (kW)'].values
    reps_values = rep_df[rep_df['Season'] == season]['LV ActivePower (kW)'].values

    plt.figure(figsize=(6,6))
    plt.boxplot([full_values, reps_values], labels=['All Scenarios', 'Representatives'], patch_artist=True,
                boxprops=dict(facecolor='skyblue', alpha=0.7),
                medianprops=dict(color='black'))
    plt.title(f'{season} Boxplot', fontsize=14)
    plt.ylabel('LV ActivePower (kW)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{season}_Boxplot.png', dpi=300)
    plt.show()

# -----------------------------------------
# Radar Chart for Metric Comparison Across Seasons
# -----------------------------------------

metrics = ['Wasserstein', 'Energy', 'CRPS', 'TVD']
num_metrics = len(metrics)
angles = [n / float(num_metrics) * 2 * pi for n in range(num_metrics)]
angles += angles[:1]

plt.figure(figsize=(8,8))
ax = plt.subplot(111, polar=True)

for season in seasons:
    values = metrics_df.loc[metrics_df['Season'] == season, metrics].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, label=season, linewidth=2, marker='o')
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics)
ax.set_title('Distance Metrics Comparison by Season', fontsize=14, y=1.2)
ax.tick_params(axis='x', pad=20)

ax.grid(True)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig('Seasonal_Metrics_Radar.png', dpi=300)
plt.show()
