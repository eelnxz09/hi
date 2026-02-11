"""
Flask Web Application for Fraud Detection
"""
from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Create uploads folder
os.makedirs('uploads', exist_ok=True)

# Load model and scaler
print("Loading model...")
try:
    model = joblib.load('fraud_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("⚠️ Model files not found. Please run train_model.py first.")
    model = None
    scaler = None

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Banking Fraud Detection System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        h1 {
            color: #667eea;
            text-align: center;
            font-size: 2.8em;
            margin-bottom: 10px;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 40px;
            font-size: 1.2em;
        }
        
        .upload-section {
            border: 3px dashed #667eea;
            border-radius: 10px;
            padding: 50px;
            text-align: center;
            margin: 30px 0;
            transition: all 0.3s;
            background: #f8f9ff;
        }
        
        .upload-section:hover {
            border-color: #764ba2;
            background: #f0f3ff;
        }
        
        .upload-section h2 {
            color: #667eea;
            margin-bottom: 20px;
        }
        
        input[type="file"] {
            margin: 20px 0;
            padding: 10px;
            font-size: 1em;
        }
        
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 50px;
            font-size: 1.2em;
            border-radius: 30px;
            cursor: pointer;
            transition: transform 0.3s;
            margin-top: 15px;
        }
        
        button:hover {
            transform: scale(1.05);
        }
        
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
            margin: 40px 0;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 30px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
        }
        
        .stat-card h3 {
            color: #667eea;
            font-size: 3em;
            margin-bottom: 10px;
        }
        
        .stat-card p {
            color: #666;
            font-size: 1.1em;
            font-weight: 500;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 30px;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        th {
            background: #667eea;
            color: white;
            padding: 18px;
            text-align: left;
            font-size: 1.05em;
        }
        
        td {
            padding: 15px 18px;
            border-bottom: 1px solid #e0e0e0;
        }
        
        tr:hover {
            background: #f8f9ff;
        }
        
        tr:last-child td {
            border-bottom: none;
        }
        
        .risk-high {
            color: #e74c3c;
            font-weight: bold;
            font-size: 1.1em;
        }
        
        .risk-medium {
            color: #f39c12;
            font-weight: bold;
            font-size: 1.1em;
        }
        
        .risk-low {
            color: #27ae60;
            font-weight: bold;
            font-size: 1.1em;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        
        .spinner {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #667eea;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error-message {
            background: #ffe0e0;
            color: #c0392b;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            display: none;
        }
        
        .section-title {
            color: #667eea;
            font-size: 1.8em;
            margin: 30px 0 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🛡️ Banking Fraud Detection System</h1>
        <p class="subtitle">AI-Powered Anomaly Detection for Secure Transactions</p>
        
        <div class="upload-section">
            <h2>📤 Upload Transaction Data</h2>
            <p style="color: #666; margin-bottom: 20px;">
                Upload a CSV file containing banking transactions
            </p>
            <input type="file" id="fileInput" accept=".csv">
            <br>
            <button onclick="analyzeTransactions()" id="analyzeBtn">
                🔍 Analyze for Fraud
            </button>
        </div>
        
        <div class="error-message" id="errorMessage"></div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p style="font-size: 1.2em; color: #667eea;">Analyzing transactions...</p>
        </div>
        
        <div id="resultsSection" style="display: none;">
            <h2 class="section-title">📊 Detection Summary</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3 id="totalTransactions">0</h3>
                    <p>Total Transactions</p>
                </div>
                <div class="stat-card">
                    <h3 id="fraudDetected" style="color: #e74c3c;">0</h3>
                    <p>Fraud Detected</p>
                </div>
                <div class="stat-card">
                    <h3 id="fraudRate">0%</h3>
                    <p>Fraud Rate</p>
                </div>
            </div>
            
            <h2 class="section-title">🚨 Flagged Transactions</h2>
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Transaction ID</th>
                            <th>Customer ID</th>
                            <th>Amount (₹)</th>
                            <th>Type</th>
                            <th>Fraud Probability</th>
                            <th>Risk Level</th>
                        </tr>
                    </thead>
                    <tbody id="tableBody">
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <script>
        function showError(message) {
            const errorDiv = document.getElementById('errorMessage');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
            setTimeout(() => {
                errorDiv.style.display = 'none';
            }, 5000);
        }
        
        async function analyzeTransactions() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            
            if (!file) {
                showError('Please select a CSV file first');
                return;
            }
            
            if (!file.name.endsWith('.csv')) {
                showError('Please upload a CSV file');
                return;
            }
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('resultsSection').style.display = 'none';
            document.getElementById('analyzeBtn').disabled = true;
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.error) {
                    showError('Error: ' + data.error);
                    return;
                }
                
                displayResults(data);
                
            } catch (error) {
                showError('Error analyzing file: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('analyzeBtn').disabled = false;
            }
        }
        
        function displayResults(data) {
            // Update statistics
            document.getElementById('totalTransactions').textContent = data.total_transactions.toLocaleString();
            document.getElementById('fraudDetected').textContent = data.fraud_detected.toLocaleString();
            document.getElementById('fraudRate').textContent = data.fraud_rate.toFixed(2) + '%';
            
            // Populate table
            const tableBody = document.getElementById('tableBody');
            tableBody.innerHTML = '';
            
            if (data.flagged_transactions.length === 0) {
                const row = tableBody.insertRow();
                const cell = row.insertCell(0);
                cell.colSpan = 6;
                cell.textContent = 'No fraudulent transactions detected';
                cell.style.textAlign = 'center';
                cell.style.padding = '30px';
                cell.style.color = '#27ae60';
                cell.style.fontSize = '1.2em';
            } else {
                data.flagged_transactions.forEach(transaction => {
                    const row = tableBody.insertRow();
                    
                    const riskClass = 'risk-' + transaction.risk_level.toLowerCase();
                    
                    row.innerHTML = `
                        <td>${transaction.transaction_id}</td>
                        <td>${transaction.customer_id}</td>
                        <td>₹${transaction.amount.toLocaleString()}</td>
                        <td>${transaction.transaction_type}</td>
                        <td>${transaction.fraud_probability.toFixed(2)}%</td>
                        <td class="${riskClass}">${transaction.risk_level}</td>
                    `;
                });
            }
            
            document.getElementById('resultsSection').style.display = 'block';
            
            // Smooth scroll to results
            document.getElementById('resultsSection').scrollIntoView({ 
                behavior: 'smooth', 
                block: 'start' 
            });
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """Home page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded transactions for fraud"""
    
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded. Please run train_model.py first.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read CSV
        df = pd.read_csv(file)
        
        # Validate required columns
        required_columns = ['transaction_id', 'customer_id', 'amount', 'timestamp', 
                          'transaction_type', 'merchant_category', 'location', 'device_type']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({
                'error': f'Missing required columns: {", ".join(missing_columns)}'
            }), 400
        
        # Feature engineering
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = df['hour'].between(22, 6).astype(int)
        
        # Customer statistics
        customer_stats = df.groupby('customer_id')['amount'].agg([
            ('avg_amount', 'mean'),
            ('std_amount', 'std'),
            ('transaction_count', 'count')
        ]).reset_index()
        
        df = df.merge(customer_stats, on='customer_id', how='left')
        df['amount_deviation'] = (df['amount'] - df['avg_amount']) / (df['std_amount'] + 1)
        
        # Encode categoricals
        df['type_encoded'] = pd.factorize(df['transaction_type'])[0]
        df['category_encoded'] = pd.factorize(df['merchant_category'])[0]
        df['device_encoded'] = pd.factorize(df['device_type'])[0]
        
        # Prepare features
        feature_columns = [
            'amount', 'hour', 'day_of_week', 'is_weekend', 'is_night',
            'amount_deviation', 'type_encoded', 'category_encoded', 
            'device_encoded', 'transaction_count'
        ]
        
        X = df[feature_columns].fillna(0)
        X_scaled = scaler.transform(X)
        
        # Predict
        predictions = model.predict(X_scaled)
        scores = model.decision_function(X_scaled)
        
        # Calculate fraud probability
        fraud_probability = 1 - (scores - scores.min()) / (scores.max() - scores.min())
        
        # Add results to dataframe
        df['is_fraud'] = (predictions == -1).astype(int)
        df['fraud_probability'] = fraud_probability * 100
        df['risk_level'] = pd.cut(
            fraud_probability, 
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        # Get flagged transactions
        flagged = df[df['is_fraud'] == 1].sort_values('fraud_probability', ascending=False)
        
        # Prepare response
        response = {
            'total_transactions': int(len(df)),
            'fraud_detected': int(flagged.shape[0]),
            'fraud_rate': float(flagged.shape[0] / len(df) * 100),
            'flagged_transactions': [
                {
                    'transaction_id': str(row['transaction_id']),
                    'customer_id': str(row['customer_id']),
                    'amount': float(row['amount']),
                    'transaction_type': str(row['transaction_type']),
                    'fraud_probability': float(row['fraud_probability']),
                    'risk_level': str(row['risk_level'])
                }
                for _, row in flagged.head(100).iterrows()
            ]
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
