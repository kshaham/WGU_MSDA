import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, SAGEConv
from torch_geometric.nn import GraphNorm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from torch_geometric.data import DataLoader
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
import os
import gc

# Set environment variables for CUDA
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# Initialize device
print("\nInitializing device...")
device = torch.device('cpu')  # Default to CPU
try:
    if torch.cuda.is_available():
        # Test CUDA with a simple operation
        test_tensor = torch.tensor([1.0], device='cuda:0')
        test_tensor = test_tensor + 1
        del test_tensor
        
        device = torch.device('cuda:0')
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
        
        # Try to clear cache with error handling
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: Could not clear CUDA cache: {e}")
    else:
        print("No GPU available. Using CPU.")
except Exception as e:
    print(f"Warning: CUDA initialization failed: {e}")
    print("Falling back to CPU")
    device = torch.device('cpu')

print(f"Using device: {device}")

# Load and preprocess the dataset
df = pd.read_csv('CT_Real_Estate_Listings_Updated.csv')

# Filter for CT single-family properties within reasonable price range
df = df[
    (df['state'] == 'CT') &  # Connecticut properties only
    (df['property_type'] == 'SINGLE_FAMILY_RESIDENTIAL') & 
    (df['price'] > 100000) &  # Minimum price threshold
    (df['price'] < 10000000)   # Maximum price threshold
].copy()

print(f"\nNumber of CT single-family properties (after filtering): {len(df)}")
print(f"Price range: ${df['price'].min():,.2f} to ${df['price'].max():,.2f}")
print(f"Median price: ${df['price'].median():,.2f}")

# Define numeric and categorical features
numeric_features = [
    'loglotArea', 'livingArea', 'latitude', 'longitude',
    'num_full_baths.x', 'num_half_baths', 'num_three_quarter_baths',
    'num_bedrooms.x', 'days_on_market', 'yearBuilt.x',
    'violent_crime', 'property_crime', 'fed_rate',
    'lagged_CPI', 'lagged_unemployment', 'volatility_value',
    'distance_to_coast', 'distance_to_new_york'
]

# Simplified categorical features for single-family homes
categorical_features = [
    'isHot', 'is_virtual_tour', 'isNew'
]

# Feature engineering first
def add_engineered_features(df):
    df = df.copy()
    
    # Price per square foot (handle zero living area)
    df['price_per_sqft'] = df['price'] / df['livingArea'].replace(0, np.nan)
    df['price_per_sqft'] = df['price_per_sqft'].replace([np.inf, -np.inf], np.nan)
    
    # Age of the property (handle missing year)
    current_year = 2024
    df['property_age'] = current_year - df['yearBuilt.x']
    df['property_age'] = df['property_age'].clip(0, 200)  # Cap age at reasonable values
    
    # Bathroom ratio (handle zero bedrooms)
    df['total_baths'] = df['num_full_baths.x'] + 0.5 * df['num_half_baths'] + 0.75 * df['num_three_quarter_baths']
    df['bath_bed_ratio'] = df['total_baths'] / df['num_bedrooms.x'].replace(0, np.nan)
    df['bath_bed_ratio'] = df['bath_bed_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Additional features for single-family homes
    df['lot_to_living_ratio'] = df['lotArea'] / df['livingArea']
    df['lot_to_living_ratio'] = df['lot_to_living_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with median
    numeric_cols = ['price_per_sqft', 'property_age', 'bath_bed_ratio', 'lot_to_living_ratio']
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    return df

# Add engineered features first
df = add_engineered_features(df)

# Update significant features list for single-family homes
significant_numeric_features = [
    'livingArea', 'lotArea', 'latitude', 'longitude',
    'num_bedrooms.x', 'yearBuilt.x',
    'distance_to_coast', 'distance_to_new_york',
    'property_age', 'bath_bed_ratio',
    'price_per_sqft', 'lot_to_living_ratio',
    'total_baths'
]

# Print feature statistics
print("\nFeature Statistics:")
for feature in significant_numeric_features:
    print(f"{feature}:")
    print(f"  Mean: {df[feature].mean():.2f}")
    print(f"  Std: {df[feature].std():.2f}")
    print(f"  Min: {df[feature].min():.2f}")
    print(f"  Max: {df[feature].max():.2f}")

# Get all numeric features including engineered ones
X_num = df[significant_numeric_features].copy()

# Handle any remaining infinite values
X_num = X_num.replace([np.inf, -np.inf], np.nan)
X_num = X_num.fillna(X_num.median())

# Scale numeric features
scaler = StandardScaler()
X_num_scaled = pd.DataFrame(
    scaler.fit_transform(X_num),
    columns=significant_numeric_features,
    index=X_num.index
)

# Encode categorical features
X_cat = pd.get_dummies(df[categorical_features], drop_first=True)

# Combine features
X = pd.concat([X_num_scaled, X_cat], axis=1)

# Convert all columns to float32 before creating tensor
X = X.astype('float32')

print("\nFeature dtypes after conversion:")
print(X.dtypes)

# Scale the target variable (price) using log transformation and standardization
y = np.log1p(df['price'])
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

print("\nTarget variable statistics:")
print(f"Original price - Mean: ${df['price'].mean():,.2f}, Std: ${df['price'].std():,.2f}")
print(f"Log price - Mean: {y.mean():.2f}, Std: {y.std():.2f}")
print(f"Scaled log price - Mean: {y_scaled.mean():.2f}, Std: {y_scaled.std():.2f}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Get coordinates for all properties
coords = df[['latitude', 'longitude']].values

# Create edges based on geographical proximity using k-nearest neighbors
def create_knn_graph(coords, k=3, batch_size=2000):
    print("Creating KNN graph...")
    edges = []
    edge_weights = []
    n_samples = len(coords)
    
    # Convert to radians for haversine distance
    coords_rad = np.radians(coords)
    
    # Use a single KNN model for all batches
    knn = NearestNeighbors(n_neighbors=k+1, metric='haversine', n_jobs=-1)
    knn.fit(coords_rad)
    
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        batch_coords = coords_rad[i:batch_end]
        
        # Calculate distances for current batch
        distances, indices = knn.kneighbors(batch_coords)
        
        # Add edges and weights for current batch
        for idx, (neighbors, dists) in enumerate(zip(indices, distances)):
            source = idx + i
            for neighbor, dist in zip(neighbors[1:], dists[1:]):
                if dist < 0.1:  # Only connect very close neighbors (about 11km)
                    edges.append([source, neighbor])
                    # Use exponential decay for weights
                    edge_weights.append(np.exp(-dist * 10))
        
        if (i + batch_size) % 5000 == 0:
            print(f"Processed {i + batch_size}/{n_samples} nodes")
    
    print("Graph creation completed")
    return torch.tensor(edges, dtype=torch.long).t().contiguous(), torch.tensor(edge_weights, dtype=torch.float)

# Create the graph
edge_index, edge_weight = create_knn_graph(coords)

# Verify edge indices are valid
print("\nVerifying edge indices...")
max_node_idx = max(edge_index[0].max().item(), edge_index[1].max().item())
if max_node_idx >= len(X_train):
    print(f"Warning: Found invalid node index {max_node_idx} (max valid index is {len(X_train)-1})")
    # Filter out invalid edges
    valid_edges = (edge_index[0] < len(X_train)) & (edge_index[1] < len(X_train))
    edge_index = edge_index[:, valid_edges]
    edge_weight = edge_weight[valid_edges]
    print(f"Removed {valid_edges.sum().item()} invalid edges")

# Move data to device with error handling
print("\nPreparing data...")
try:
    # Create tensors on CPU first with explicit dtype
    print("Creating tensors...")
    node_features = torch.tensor(X_train.values, dtype=torch.float32, device='cpu')
    y_tensor = torch.tensor(y_train.values, dtype=torch.float32, device='cpu')
    
    # Ensure edge_index is in the correct format and dtype
    edge_index = edge_index.to(torch.long)
    edge_weight = edge_weight.to(torch.float32)
    
    # Verify edge_index format
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must be of shape [2, num_edges], got {edge_index.shape}")
    
    # Create train/test split
    num_nodes = len(X_train)
    indices = torch.randperm(num_nodes, device='cpu')
    train_size = int(0.8 * num_nodes)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device='cpu')
    train_mask[train_indices] = True
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device='cpu')
    test_mask[test_indices] = True
    
    print("\nData shapes:")
    print(f"Node features: {node_features.shape}")
    print(f"Edge index: {edge_index.shape}")
    print(f"Edge weights: {edge_weight.shape}")
    print(f"Target: {y_tensor.shape}")
    print(f"Training samples: {train_mask.sum().item()}")
    print(f"Testing samples: {test_mask.sum().item()}")
    
    # Check data integrity
    print("\nChecking data integrity...")
    try:
        assert not torch.isnan(node_features).any(), "NaN values found in node features"
        assert not torch.isinf(node_features).any(), "Infinite values found in node features"
        assert not torch.isnan(y_tensor).any(), "NaN values found in target tensor"
        assert not torch.isinf(y_tensor).any(), "Infinite values found in target tensor"
        assert not torch.isnan(edge_weight).any(), "NaN values found in edge weights"
        assert not torch.isinf(edge_weight).any(), "Infinite values found in edge weights"
        assert edge_index.dtype == torch.long, f"edge_index must be long type, got {edge_index.dtype}"
        assert (edge_index >= 0).all(), "Negative indices found in edge_index"
        assert (edge_index < num_nodes).all(), "Indices out of bounds in edge_index"
        print("Data integrity check passed")
    except AssertionError as e:
        print(f"Data integrity check failed: {e}")
        raise

    # Move all data to GPU in smaller chunks
    if device.type == 'cuda':
        print("\nMoving data to GPU...")
        try:
            # Move data in smaller chunks to avoid memory issues
            chunk_size = 1000  # Adjust based on your GPU memory
            
            # Move node features in chunks
            for i in range(0, node_features.size(0), chunk_size):
                node_features[i:i+chunk_size] = node_features[i:i+chunk_size].to(device)
            print("Node features moved to GPU")
            
            # Move edge index and weights
            edge_index = edge_index.to(device)
            edge_weight = edge_weight.to(device)
            print("Edge data moved to GPU")
            
            # Move target tensor
            y_tensor = y_tensor.to(device)
            print("Target moved to GPU")
            
            # Move masks to the same device
            train_mask = train_mask.to(device)
            test_mask = test_mask.to(device)
            print("Masks moved to GPU")
            
            print("\nAll data successfully moved to GPU")
            print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f}MB / {torch.cuda.memory_reserved() / 1024**2:.1f}MB")
            
        except RuntimeError as e:
            print(f"Error moving data to GPU: {e}")
            print("Falling back to CPU")
            device = torch.device('cpu')
            # Keep data on CPU

except Exception as e:
    print(f"Error in data preparation: {e}")
    raise

# Simplify model for faster training
class GraphSAGEModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, 1)
        self.dropout = torch.nn.Dropout(0.2)
        self.linear = torch.nn.Linear(hidden_channels, hidden_channels)
        
    def forward(self, x, edge_index):
        # Ensure edge_index is long type
        if edge_index.dtype != torch.long:
            edge_index = edge_index.long()
        
        # First layer
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.elu(x1)
        x1 = self.dropout(x1)
        
        # Second layer with residual connection
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.elu(x2)
        x2 = self.dropout(x2)
        x2 = x2 + self.linear(x1)  # Residual connection
        
        # Third layer with residual connection
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.elu(x3)
        x3 = self.dropout(x3)
        x3 = x3 + self.linear(x2)  # Residual connection
        
        # Final layer
        x4 = self.conv4(x3, edge_index)
        return x4.squeeze()

# Initialize model and optimizer
print("\nInitializing model...")
model = GraphSAGEModel(num_node_features=node_features.size(1)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
criterion = torch.nn.MSELoss()

# Move data to device
print("\nMoving data to device...")
node_features = node_features.to(device)
edge_index = edge_index.to(device)
edge_weight = edge_weight.to(device)
y_tensor = y_tensor.to(device)
train_mask = train_mask.to(device)
test_mask = test_mask.to(device)

# Training loop
print("\nTraining model...")
best_val_loss = float('inf')
best_model_state = None
patience = 15
counter = 0
train_losses = []
val_losses = []

try:
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = model(node_features, edge_index)
        loss = criterion(out[train_mask], y_tensor[train_mask])
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(node_features, edge_index)
            val_loss = criterion(val_out[test_mask], y_tensor[test_mask])
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
        
        # Early stopping
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Store losses
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
            
except Exception as e:
    print(f"Training interrupted by error: {e}")
    if device.type == 'cuda':
        torch.cuda.empty_cache()

# Load best model if available
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("Loaded best model state")

# Feature Importance Analysis
print("\nCalculating feature importance...")
model.eval()
with torch.no_grad():
    # Get baseline predictions
    baseline_pred = model(node_features, edge_index)
    baseline_loss = criterion(baseline_pred[test_mask], y_tensor[test_mask])
    
    feature_importance = {}
    for i, feature_name in enumerate(significant_numeric_features):
        # Create perturbed features
        perturbed_features = node_features.clone()
        # Add noise to the feature
        noise = torch.randn_like(perturbed_features[:, i]) * 0.1
        perturbed_features[:, i] += noise
        
        # Get predictions with perturbed features
        perturbed_pred = model(perturbed_features, edge_index)
        perturbed_loss = criterion(perturbed_pred[test_mask], y_tensor[test_mask])
        
        # Calculate importance as the increase in loss
        importance = (perturbed_loss - baseline_loss).item()
        feature_importance[feature_name] = importance

# Sort and display feature importance
print("\nFeature Importance (higher values indicate more important features):")
sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
for feature, importance in sorted_features:
    print(f"{feature}: {importance:.4f}")

# Plot feature importance
plt.figure(figsize=(12, 6))
features, importances = zip(*sorted_features)
plt.barh(features, importances)
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# Error Analysis
print("\nAnalyzing prediction errors...")
with torch.no_grad():
    predictions = model(node_features, edge_index)
    y_pred = predictions[test_mask].cpu().numpy()
    y_true = y_tensor[test_mask].cpu().numpy()

# Convert predictions back to original scale
y_pred_original = np.expm1(y_pred)
y_test_original = np.expm1(y_true)

# Calculate absolute percentage errors
absolute_errors = np.abs(y_pred_original - y_test_original)
percentage_errors = (absolute_errors / y_test_original) * 100

# Print error statistics
print(f"\nError Analysis:")
print(f"Mean Absolute Percentage Error: {np.mean(percentage_errors):.2f}%")
print(f"Median Absolute Percentage Error: {np.median(percentage_errors):.2f}%")
print(f"90th Percentile Absolute Percentage Error: {np.percentile(percentage_errors, 90):.2f}%")

# Plot error distribution
plt.figure(figsize=(12, 6))
plt.hist(percentage_errors, bins=50, alpha=0.7)
plt.title('Distribution of Percentage Errors')
plt.xlabel('Percentage Error (%)')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# Calculate metrics first
mae = mean_absolute_error(y_test_original, y_pred_original)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
r2 = r2_score(y_test_original, y_pred_original)
mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
median_ape = np.median(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
r2_adj = 1 - (1 - r2) * (len(y_test_original) - 1) / (len(y_test_original) - X_test.shape[1] - 1)

print("\nModel Performance Metrics:")
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"R-squared (R²) Score: {r2:.4f}")
print(f"Adjusted R-squared (R²_adj): {r2_adj:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Median Absolute Percentage Error: {median_ape:.2f}%")

# Create a single figure with all plots
fig, axes = plt.subplots(2, 3, figsize=(20, 15))

# Training history
axes[0, 0].plot(train_losses, label='Training Loss')
axes[0, 0].plot(val_losses, label='Validation Loss')
axes[0, 0].set_title('Model Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Actual vs Predicted scatter plot
axes[0, 1].scatter(y_test_original, y_pred_original, alpha=0.5)
axes[0, 1].plot([y_test_original.min(), y_test_original.max()], 
                [y_test_original.min(), y_test_original.max()], 'r--')
axes[0, 1].set_xlabel('Actual Price ($)')
axes[0, 1].set_ylabel('Predicted Price ($)')
axes[0, 1].set_title('Actual vs Predicted Property Prices')
axes[0, 1].grid(True)

# Error distribution
axes[0, 2].hist(percentage_errors, bins=50, alpha=0.7)
axes[0, 2].set_title('Distribution of Percentage Errors')
axes[0, 2].set_xlabel('Percentage Error (%)')
axes[0, 2].set_ylabel('Count')
axes[0, 2].grid(True)

# Feature importance
features, importances = zip(*sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True))
axes[1, 0].barh(features, importances)
axes[1, 0].set_title('Feature Importance')
axes[1, 0].set_xlabel('Importance Score')

# Create scatter plots for actual and predicted prices
test_indices = np.where(test_mask.cpu().numpy())[0]

# Actual prices
scatter1 = axes[1, 1].scatter(df.iloc[test_indices].longitude, 
                            df.iloc[test_indices].latitude, 
                            c=df.iloc[test_indices].price, 
                            cmap='YlOrRd', alpha=0.7)
plt.colorbar(scatter1, ax=axes[1, 1], label='Actual Price ($)')
axes[1, 1].set_title('Actual Property Prices')
axes[1, 1].set_xlabel('Longitude')
axes[1, 1].set_ylabel('Latitude')
axes[1, 1].grid(True)

# Predicted prices
scatter2 = axes[1, 2].scatter(df.iloc[test_indices].longitude, 
                            df.iloc[test_indices].latitude, 
                            c=y_pred_original, 
                            cmap='YlOrRd', alpha=0.7)
plt.colorbar(scatter2, ax=axes[1, 2], label='Predicted Price ($)')
axes[1, 2].set_title('Predicted Property Prices')
axes[1, 2].set_xlabel('Longitude')
axes[1, 2].set_ylabel('Latitude')
axes[1, 2].grid(True)

# Adjust layout and display
plt.tight_layout()
plt.show()

# Clear GPU memory
if device.type == 'cuda':
    torch.cuda.empty_cache() 