import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier

mcc_scorer = make_scorer(matthews_corrcoef)


def compute_sharpe_ratio(y_true, y_pred, stock_data, risk_free_rate=0.02 / 252):
    """Calculate the Sharpe Ratio based on model predictions."""

    stock_data['Predicted_Signal'] = y_pred
    stock_data['Next_Close'] = stock_data['Close'].shift(-1)  # Next day's close price

    # Compute returns only for Buy (1) signals
    stock_data['Return'] = np.where(stock_data['Predicted_Signal'] == 1,
                                    (stock_data['Next_Close'] - stock_data['Close']) / stock_data['Close'],
                                    0)  # No trade = 0 return

    mean_return = stock_data['Return'].mean()
    volatility = stock_data['Return'].std()

    sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility != 0 else 0
    return sharpe_ratio


# Update the Sharpe Ratio scorer
sharpe_scorer = make_scorer(compute_sharpe_ratio, greater_is_better=True, needs_proba=False)

sharpe_scorer = make_scorer(sharpe_ratio, greater_is_better=True)


pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.expand_frame_repr', False)  # Prevent line wrapping

stock_data = yf.download("AAPL", start="2015-01-01", end="2025-01-31", progress=False)
stock_data = stock_data.asfreq('B')  # Ensure business days only


# Add Additional Technical Indicators
stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
stock_data['SMA_50'] = stock_data['Close'].ewm(span=50, adjust=False).mean()

stock_data['BB_std_20'] = stock_data['Close'].rolling(window=20).std()
stock_data['BB_upper_20'] = stock_data['SMA_20'] + (2 * stock_data['Close'].rolling(window=20).std().squeeze())
stock_data['BB_lower_20'] = stock_data['SMA_20'] - (2 * stock_data['Close'].rolling(window=20).std().squeeze())

# stock_data['BB_std_50'] = stock_data['Close'].rolling(window=50).std()
# stock_data['BB_upper_50'] = stock_data['SMA_50'] + (2 * stock_data['Close'].rolling(window=50).std().squeeze())
# stock_data['BB_lower_50'] = stock_data['SMA_50'] - (2 * stock_data['Close'].rolling(window=50).std().squeeze())
#
# # Generate Bollinger Band Trading Signals**
# stock_data['BB_Signal'] = np.where(stock_data['Close'] < stock_data['BB_lower_20'], 1,  # Buy Signal
#                           np.where(stock_data['Close'] > stock_data['BB_upper_20'], -1, 0))  # Sell / Hold




# Compute MACD
short_ema = stock_data['Close'].ewm(span=12, adjust=False).mean()
long_ema = stock_data['Close'].ewm(span=26, adjust=False).mean()
stock_data['MACD'] = short_ema - long_ema

# Compute Signal Line (9-day EMA of MACD)
stock_data['Signal_Line'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()

# Compute MACD Histogram
stock_data['MACD_Histogram'] = stock_data['MACD'] - stock_data['Signal_Line']
#

# Compute Stochastic Oscillator
low_14 = stock_data['Low'].rolling(window=14).min()
high_14 = stock_data['High'].rolling(window=14).max()
stock_data['Stochastic_%K'] = ((stock_data['Close'] - low_14) / (high_14 - low_14)) * 100

# Compute Stochastic %D (3-day moving average of %K)
stock_data['Stochastic_%D'] = stock_data['Stochastic_%K'].rolling(window=3).mean()


# Relative Strength Index (RSI)
delta = stock_data['Close'].diff(1)
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
stock_data['RSI'] = 100 - (100 / (1 + rs))

# Compute features
stock_data.dropna(inplace=True)  # Drop NaNs **before feature selection**

#target
threshold = stock_data['Close'].pct_change().rolling(window=5).std()  # Use volatility as a buffer
#

stock_data['Volume_Change'] = stock_data['Volume'].pct_change()
stock_data['VWAP'] = (stock_data['Close'] * stock_data['Volume']).cumsum() / stock_data['Volume'].cumsum()
stock_data['Volatility'] = stock_data['Close'].pct_change().rolling(window=10).std()



threshold = stock_data['Close'].pct_change().rolling(window=5).std()
stock_data['Target'] = np.where(stock_data['Close'].shift(-1) > (stock_data['Close'] + threshold), 1, 0)



#print(stock_data.head(100))




features = ['SMA_20', 'SMA_50', 'BB_std_20', 'BB_upper_20', 'BB_lower_20', 'RSI', 'MACD', 'Signal_Line',
                'MACD_Histogram', 'Stochastic_%K', 'Stochastic_%D', 'Volume_Change','VWAP', 'Volatility' ]

correlation_matrix = stock_data[features].corr()
high_corr_features = correlation_matrix[correlation_matrix > 0.9].stack().index.tolist()
features = [col for col in features if col not in high_corr_features]

# Remove NaN values from the dataset at once
stock_data = stock_data.dropna()

# Define features and target again after dropping NaNs
X = stock_data[features]
y = stock_data['Target']

# Now split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # Use same scaler on test set


# Define Models
models = {
    "XGBoost": XGBClassifier(random_state=42),
    "MLP": MLPClassifier(random_state=42, max_iter=5000),
    "RandomForest": RandomForestClassifier(random_state=42, class_weight='balanced')
}

# Define Hyperparameter Grids
param_grids = {
    "XGBoost": {
        'n_estimators': [50, 100, 150, 200, 300],  # Expanding search
        'max_depth': [2, 3, 4, 5],  # Add even shallower trees
        'learning_rate': [0.005, 0.01, 0.05, 0.1],  # More refined learning rates
        'subsample': [0.5, 0.6, 0.7, 0.8],  # Introduce more randomness
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8],  # More feature selection variations
        'gamma': [0.1, 0.3, 0.5, 1, 2],  # Increase pruning strength
        'min_child_weight': [3, 5, 7, 10],  # Prevent overfitting by avoiding small leaf nodes
        'lambda': [1, 5, 10, 20],  # L2 regularization
        'alpha': [0, 1, 5, 10]  # L1 regularization
    },

    "RandomForest": {
        'n_estimators': [50, 100, 150, 300],  # More diverse number of trees
        'max_depth': [3, 5, 7, 10],  # More depth choices
        'min_samples_split': [5, 10, 20, 30],  # Increase required samples per split
        'min_samples_leaf': [2, 4, 8, 12],  # Larger leaf nodes to prevent overfitting
        'bootstrap': [True, False],  # Test impact of full sample training
        'criterion': ['gini', 'entropy'],  # Compare different split criteria
        'max_features': ['sqrt', 'log2', None]  # Experiment with different feature selection strategies
    },

    "MLP": {
        'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100), (250,), (500,), (200, 100, 50)],
        # Added deeper networks
        'activation': ['relu', 'tanh'],  # Test different activation functions
        'solver': ['adam'],
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1],  # Increase regularization strength
        'learning_rate': ['constant', 'adaptive', 'invscaling'],  # More LR variations
        'early_stopping': [True],  # Prevent overfitting
        'batch_size': [16, 32, 64]  # Control mini-batch size
    }

}

# Function to run full GridSearchCV with MCC
def tune_model(name, model, param_grid, X_train, y_train):
    print(f"\nRunning Grid Search for {name} using MCC...")
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,  # Reduce CV folds to speed up tuning
        scoring={'MCC': mcc_scorer, 'Sharpe': sharpe_scorer},  # Use MCC as the metric
        n_jobs=-1,  # Use all available cores
        verbose=10
    )
    search.fit(X_train, y_train)
    return name, search.best_estimator_, search.best_params_, search.best_score_


# Run searches in parallel
results = Parallel(n_jobs=3)(
    delayed(tune_model)(name, models[name], param_grids[name], X_train, y_train)
    for name in models
)

# Store selected models
selected_models = {}

# Evaluate Models & Prevent Overfitting
for name, best_model, best_params, best_score in results:
    print(f"\nBest {name} Hyperparameters: {best_params}")
    print(f"Best {name} MCC (Cross-Validation): {best_score:.2f}")

    # Predict on training set
    y_train_pred = best_model.predict(X_train)
    train_mcc = matthews_corrcoef(y_train, y_train_pred)

    # Predict on test set
    y_test_pred = best_model.predict(X_test)
    test_mcc = matthews_corrcoef(y_test, y_test_pred)

    print(f"{name} Training MCC: {train_mcc:.2f}")
    print(f"{name} Test MCC: {test_mcc:.2f}")

    # Compute train/test ratio (prevent overfitting)
    mcc_ratio = test_mcc / train_mcc if train_mcc != 0 else 0

    if mcc_ratio >= 0:
        print(f"✅ {name} Passed Overfitting Test! (MCC Ratio: {mcc_ratio:.2f})")
        selected_models[name] = best_model  # Store the valid model
    else:
        print(f"❌ {name} Overfit Detected! Skipping Model (MCC Ratio: {mcc_ratio:.2f})")

# Train and Evaluate the Ensemble Model
if 'XGBoost' in selected_models and 'RandomForest' in selected_models:
    ensemble_model = VotingClassifier(estimators=[
        ('xgb', selected_models['XGBoost']),
        ('rf', selected_models['RandomForest'])
    ], voting='soft', n_jobs=-1)

    ensemble_model.fit(X_train, y_train)
    y_pred_ensemble = ensemble_model.predict(X_test)
    ensemble_mcc = matthews_corrcoef(y_test, y_pred_ensemble)
    print(f"\n🚀 Ensemble Model MCC: {ensemble_mcc:.2f}")

# Print selected models
print("\n✅ Final Selected Models:")
for model_name in selected_models:
    print(f"✔ {model_name}")

