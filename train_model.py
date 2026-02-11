import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

# Generate data
np.random.seed(42)
transactions = []
customers = [f"CUST{i:04d}" for i in range(1, 201)]

for i in range(850):
    transactions.append({
        'transaction_id': f"TXN{i+1:06d}",
        'customer_id': random.choice(customers),
        'amount': round(max(100, np.random.normal(5000, 2000)), 2),
        'timestamp': datetime.now() - timedelta(days=random.randint(0, 90)),
        'transaction_type': random.choice(['Purchase', 'Transfer', 'Withdrawal']),
        'merchant_category': random.choice(['Retail', 'Food', 'Travel', 'Online']),
        'location': 'Mumbai',
        'device_type': 'Mobile',
        'is_fraud': 0
    })

for i in range(150):
    transactions.append({
        'transaction_id': f"TXN{851+i:06d}",
        'customer_id': random.choice(customers),
        'amount': round(np.random.uniform(20000, 100000), 2),
        'timestamp': datetime.now() - timedelta(days=random.randint(0, 90)),
        'transaction_type': 'Transfer',
        'merchant_category': 'Online',
        'location': 'Unknown',
        'device_type': 'Desktop',
        'is_fraud': 1
    })

df = pd.DataFrame(transactions)
df.to_csv('sample_transactions.csv', index=False)

# Train
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = 0
df['is_night'] = 0

stats = df.groupby('customer_id')['amount'].agg(['mean', 'std', 'count']).reset_index()
stats.columns = ['customer_id', 'avg_amount', 'std_amount', 'transaction_count']
df = df.merge(stats, on='customer_id')
df['amount_deviation'] = (df['amount'] - df['avg_amount']) / (df['std_amount'] + 1)
df['type_encoded'] = pd.factorize(df['transaction_type'])[0]
df['category_encoded'] = pd.factorize(df['merchant_category'])[0]
df['device_encoded'] = pd.factorize(df['device_type'])[0]

features = ['amount', 'hour', 'day_of_week', 'is_weekend', 'is_night',
            'amount_deviation', 'type_encoded', 'category_encoded', 
            'device_encoded', 'transaction_count']

X = df[features].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(n_estimators=100, contamination=0.15, random_state=42)
model.fit(X_scaled)

joblib.dump(model, 'fraud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model trained!")
