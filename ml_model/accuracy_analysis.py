import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from train_model import fetch_stock_features, label_data
import matplotlib.pyplot as plt

def analyze_model_accuracy(ticker):
    """Comprehensive accuracy analysis for a trained model"""
    print(f"\n🔍 DETAILED ACCURACY ANALYSIS FOR {ticker}")
    print("=" * 60)
    
    try:
        # Load model components
        model = joblib.load(f"{ticker}_model.pkl")
        scaler = joblib.load(f"{ticker}_scaler.pkl")
        feature_columns = joblib.load(f"{ticker}_features.pkl")
        
        # Get fresh data for testing
        df = fetch_stock_features(ticker)
        df = label_data(df)
        
        # Prepare features
        features = df[feature_columns]
        labels = df["Label"]
        X_scaled = scaler.transform(features)
        
        # Make predictions on all data
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        
        print(f"📊 OVERALL ACCURACY: {accuracy:.1%}")
        print(f"📈 DATASET SIZE: {len(labels)} samples")
        print(f"📅 DATA PERIOD: Last 2 years")
        
        # Label distribution
        print(f"\n📋 ACTUAL LABEL DISTRIBUTION:")
        label_counts = labels.value_counts()
        for label, count in label_counts.items():
            percentage = count / len(labels) * 100
            print(f"   {label.upper()}: {count} samples ({percentage:.1f}%)")
        
        # Prediction distribution
        print(f"\n🎯 PREDICTED LABEL DISTRIBUTION:")
        pred_counts = pd.Series(predictions).value_counts()
        for label, count in pred_counts.items():
            percentage = count / len(predictions) * 100
            print(f"   {label.upper()}: {count} samples ({percentage:.1f}%)")
        
        # Detailed classification report
        print(f"\n📈 DETAILED PERFORMANCE:")
        print(classification_report(labels, predictions))
        
        # Confusion Matrix
        print(f"\n🔢 CONFUSION MATRIX:")
        cm = confusion_matrix(labels, predictions)
        cm_df = pd.DataFrame(cm, 
                           index=['Actual_' + str(i) for i in model.classes_], 
                           columns=['Pred_' + str(i) for i in model.classes_])
        print(cm_df)
        
        # Confidence analysis
        max_probs = probabilities.max(axis=1)
        avg_confidence = max_probs.mean()
        print(f"\n🎯 CONFIDENCE ANALYSIS:")
        print(f"   Average Confidence: {avg_confidence:.1%}")
        print(f"   High Confidence (>80%): {(max_probs > 0.8).sum()} samples ({(max_probs > 0.8).mean():.1%})")
        print(f"   Low Confidence (<60%): {(max_probs < 0.6).sum()} samples ({(max_probs < 0.6).mean():.1%})")
        
        # Accuracy by confidence level
        high_conf_mask = max_probs > 0.8
        if high_conf_mask.sum() > 0:
            high_conf_accuracy = accuracy_score(labels[high_conf_mask], predictions[high_conf_mask])
            print(f"   Accuracy on High Confidence Predictions: {high_conf_accuracy:.1%}")
        
        # Random baseline comparison
        random_accuracy = label_counts.max() / len(labels)  # Most frequent class baseline
        print(f"\n📊 BASELINE COMPARISONS:")
        print(f"   Random Baseline (most frequent class): {random_accuracy:.1%}")
        print(f"   Model Improvement over Baseline: {(accuracy - random_accuracy)*100:.1f} percentage points")
        
        # Real-world context
        print(f"\n💡 REAL-WORLD CONTEXT:")
        if accuracy > 0.6:
            print(f"   ✅ {accuracy:.1%} accuracy is GOOD for stock prediction")
        else:
            print(f"   ⚠️ {accuracy:.1%} accuracy needs improvement")
            
        print(f"   📚 Professional fund managers typically achieve 50-60% accuracy")
        print(f"   🎯 Your model: {accuracy:.1%} - {'ABOVE' if accuracy > 0.55 else 'BELOW'} professional level")
        
        return accuracy, avg_confidence, len(labels)
        
    except Exception as e:
        print(f"❌ Error analyzing {ticker}: {e}")
        return None, None, None

def compare_all_models():
    """Compare accuracy across all trained models"""
    print(f"\n🏆 MODEL COMPARISON SUMMARY")
    print("=" * 80)
    
    tickers = ["AAPL", "NVDA", "AMZN", "GOOGL", "TSLA"]
    results = []
    
    for ticker in tickers:
        accuracy, confidence, samples = analyze_model_accuracy(ticker)
        if accuracy:
            results.append({
                'Ticker': ticker,
                'Accuracy': f"{accuracy:.1%}",
                'Avg_Confidence': f"{confidence:.1%}",
                'Samples': samples,
                'Grade': 'A' if accuracy > 0.7 else 'B' if accuracy > 0.6 else 'C'
            })
    
    # Summary table
    if results:
        print(f"\n📊 FINAL SUMMARY TABLE:")
        df_results = pd.DataFrame(results)
        print(df_results.to_string(index=False))
        
        avg_accuracy = sum([float(r['Accuracy'].strip('%'))/100 for r in results]) / len(results)
        print(f"\n🎯 PORTFOLIO AVERAGE ACCURACY: {avg_accuracy:.1%}")
        
        if avg_accuracy > 0.65:
            print("🏆 EXCELLENT: Your models perform well above market standards!")
        elif avg_accuracy > 0.55:
            print("✅ GOOD: Your models outperform typical professional managers!")
        else:
            print("⚠️ NEEDS IMPROVEMENT: Consider tuning hyperparameters or adding features")

if __name__ == "__main__":
    compare_all_models()
