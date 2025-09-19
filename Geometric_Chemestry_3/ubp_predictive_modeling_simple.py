#!/usr/bin/env python3
"""
Simplified UBP-Enhanced Predictive Modeling for Materials Discovery
Phase 6: Generate predictive models using available UBP features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import json
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main execution function"""
    
    print("="*80)
    print("UBP-ENHANCED PREDICTIVE MODELING (SIMPLIFIED)")
    print("="*80)
    print("Phase 6: Generating Predictive Models for UBP Features")
    print()
    
    # Load data
    print("Loading UBP-encoded data...")
    try:
        df = pd.read_csv("ubp_encoded_inorganic_materials.csv")
        print(f"✅ Loaded {len(df)} UBP-encoded materials with {len(df.columns)} features")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # Define prediction targets
    prediction_targets = {
        'ubp_quality_score': 'regression',
        'nrci_calculated': 'regression', 
        'primary_realm': 'classification',
        'system_coherence': 'regression',
        'total_resonance_potential': 'regression'
    }
    
    print("\\nStep 1: Feature Engineering...")
    
    # Get all numeric features except targets
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    feature_columns = [col for col in numeric_columns if col not in prediction_targets.keys()]
    
    print(f"  Using {len(feature_columns)} numeric features")
    
    # Create feature matrix
    X = df[feature_columns].fillna(df[feature_columns].median())
    
    print("\\nStep 2: Model Training...")
    
    results = {}
    
    for target_name, target_type in prediction_targets.items():
        if target_name not in df.columns:
            print(f"  ⚠️  {target_name} not available in dataset")
            continue
            
        print(f"  Training models for {target_name}...")
        
        # Prepare target
        y = df[target_name].copy()
        
        if target_type == 'classification':
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))
        
        # Remove NaN values
        valid_mask = ~pd.isna(y)
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        if len(y_valid) < 20:
            print(f"    ⚠️  Insufficient data for {target_name}")
            continue
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_valid, y_valid, test_size=0.2, random_state=42
        )
        
        # Train model
        if target_type == 'regression':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            metrics = {
                'train_r2': float(train_r2),
                'test_r2': float(test_r2),
                'train_rmse': float(train_rmse),
                'test_rmse': float(test_rmse)
            }
            
            print(f"    ✅ Random Forest: R² = {test_r2:.3f}")
            
        else:  # classification
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Metrics
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            
            metrics = {
                'train_accuracy': float(train_acc),
                'test_accuracy': float(test_acc)
            }
            
            print(f"    ✅ Random Forest: Accuracy = {test_acc:.3f}")
        
        # Feature importance
        feature_importance = dict(zip(feature_columns, model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        results[target_name] = {
            'target_type': target_type,
            'metrics': metrics,
            'predictions': {
                'y_test_true': y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test),
                'y_test_pred': y_test_pred.tolist() if hasattr(y_test_pred, 'tolist') else list(y_test_pred)
            },
            'top_features': top_features,
            'sample_count': len(y_valid)
        }
    
    print("\\nStep 3: UBP Analysis...")
    
    # Analyze UBP-specific metrics
    ubp_analysis = {}
    
    # NRCI analysis
    if 'nrci_calculated' in df.columns:
        nrci_values = df['nrci_calculated'].dropna()
        nrci_target = 0.999999
        achievement_rate = np.sum(nrci_values >= nrci_target) / len(nrci_values) * 100
        
        ubp_analysis['nrci'] = {
            'target': nrci_target,
            'achievement_rate': float(achievement_rate),
            'mean': float(nrci_values.mean()),
            'std': float(nrci_values.std())
        }
        
        print(f"  NRCI Target Achievement: {achievement_rate:.1f}%")
    
    # Quality score analysis
    if 'ubp_quality_score' in df.columns:
        quality_scores = df['ubp_quality_score'].dropna()
        high_quality_rate = np.sum(quality_scores > 0.8) / len(quality_scores) * 100
        
        ubp_analysis['quality'] = {
            'high_quality_rate': float(high_quality_rate),
            'mean': float(quality_scores.mean()),
            'std': float(quality_scores.std())
        }
        
        print(f"  High Quality Materials: {high_quality_rate:.1f}%")
    
    # Realm distribution
    if 'primary_realm' in df.columns:
        realm_dist = df['primary_realm'].value_counts().to_dict()
        ubp_analysis['realms'] = realm_dist
        
        print(f"  Dominant Realm: {max(realm_dist.items(), key=lambda x: x[1])[0]}")
    
    print("\\nStep 4: Creating Visualizations...")
    
    # Create visualization
    try:
        n_targets = len(results)
        if n_targets > 0:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            plot_idx = 0
            
            # Plot 1: Model Performance
            target_names = []
            scores = []
            
            for target_name, result in results.items():
                target_names.append(target_name.replace('_', ' ').title())
                if result['target_type'] == 'regression':
                    scores.append(result['metrics']['test_r2'])
                else:
                    scores.append(result['metrics']['test_accuracy'])
            
            colors = ['green' if s > 0.7 else 'orange' if s > 0.5 else 'red' for s in scores]
            bars = axes[plot_idx].bar(range(len(target_names)), scores, color=colors)
            axes[plot_idx].set_xlabel('UBP Target')
            axes[plot_idx].set_ylabel('Model Score')
            axes[plot_idx].set_title('UBP Prediction Model Performance')
            axes[plot_idx].set_xticks(range(len(target_names)))
            axes[plot_idx].set_xticklabels(target_names, rotation=45, ha='right')
            axes[plot_idx].grid(True, alpha=0.3)
            
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                axes[plot_idx].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{score:.3f}', ha='center', va='bottom')
            
            plot_idx += 1
            
            # Plot 2: NRCI Prediction (if available)
            if 'nrci_calculated' in results:
                nrci_result = results['nrci_calculated']
                y_true = nrci_result['predictions']['y_test_true']
                y_pred = nrci_result['predictions']['y_test_pred']
                
                axes[plot_idx].scatter(y_true, y_pred, alpha=0.6, s=30)
                
                min_val = min(min(y_true), min(y_pred))
                max_val = max(max(y_true), max(y_pred))
                axes[plot_idx].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                
                axes[plot_idx].set_xlabel('Actual NRCI')
                axes[plot_idx].set_ylabel('Predicted NRCI')
                axes[plot_idx].set_title(f'NRCI Prediction\\nR² = {nrci_result["metrics"]["test_r2"]:.3f}')
                axes[plot_idx].grid(True, alpha=0.3)
                
                plot_idx += 1
            
            # Plot 3: UBP Quality Distribution
            if 'ubp_quality_score' in df.columns:
                quality_scores = df['ubp_quality_score'].dropna()
                axes[plot_idx].hist(quality_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                axes[plot_idx].axvline(0.8, color='red', linestyle='--', label='High Quality Threshold')
                axes[plot_idx].set_xlabel('UBP Quality Score')
                axes[plot_idx].set_ylabel('Frequency')
                axes[plot_idx].set_title('UBP Quality Score Distribution')
                axes[plot_idx].legend()
                axes[plot_idx].grid(True, alpha=0.3)
                
                plot_idx += 1
            
            # Plot 4: Realm Distribution
            if 'primary_realm' in df.columns:
                realm_dist = df['primary_realm'].value_counts()
                axes[plot_idx].pie(realm_dist.values, labels=realm_dist.index, autopct='%1.1f%%', startangle=90)
                axes[plot_idx].set_title('UBP Realm Distribution')
                
                plot_idx += 1
            
            # Plot 5: Feature Importance (best model)
            if results:
                best_target = max(results.items(), key=lambda x: x[1]['metrics'].get('test_r2', x[1]['metrics'].get('test_accuracy', 0)))
                top_features = best_target[1]['top_features']
                
                feature_names = [name.replace('_', ' ') for name, _ in top_features]
                importances = [importance for _, importance in top_features]
                
                axes[plot_idx].barh(range(len(feature_names)), importances, color='lightcoral')
                axes[plot_idx].set_xlabel('Feature Importance')
                axes[plot_idx].set_ylabel('Features')
                axes[plot_idx].set_title(f'Top Features for {best_target[0].replace("_", " ").title()}')
                axes[plot_idx].set_yticks(range(len(feature_names)))
                axes[plot_idx].set_yticklabels(feature_names)
                axes[plot_idx].grid(True, alpha=0.3)
                
                plot_idx += 1
            
            # Plot 6: UBP Metrics Summary
            if ubp_analysis:
                metrics = []
                values = []
                
                if 'nrci' in ubp_analysis:
                    metrics.append('NRCI Achievement')
                    values.append(ubp_analysis['nrci']['achievement_rate'])
                
                if 'quality' in ubp_analysis:
                    metrics.append('High Quality Rate')
                    values.append(ubp_analysis['quality']['high_quality_rate'])
                
                if metrics:
                    axes[plot_idx].bar(metrics, values, color=['blue', 'green'])
                    axes[plot_idx].set_ylabel('Percentage (%)')
                    axes[plot_idx].set_title('UBP Achievement Metrics')
                    axes[plot_idx].grid(True, alpha=0.3)
                    
                    for i, v in enumerate(values):
                        axes[plot_idx].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
                
                plot_idx += 1
            
            # Hide unused subplots
            for i in range(plot_idx, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig('ubp_predictive_modeling_results.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("  ✅ Created UBP prediction visualizations")
    
    except Exception as e:
        print(f"  ⚠️  Error creating visualizations: {e}")
    
    print("\\nStep 5: Saving Results...")
    
    # Save results
    final_results = {
        'model_results': results,
        'ubp_analysis': ubp_analysis,
        'discovery_insights': {
            'high_nrci_materials': ubp_analysis.get('nrci', {}).get('achievement_rate', 0),
            'high_quality_materials': ubp_analysis.get('quality', {}).get('high_quality_rate', 0),
            'dominant_realm': max(ubp_analysis.get('realms', {'unknown': 1}).items(), key=lambda x: x[1])[0]
        },
        'optimization_recommendations': {
            'nrci_optimization': 'Focus on materials with NRCI ≥ 0.999999',
            'quality_enhancement': 'Target UBP quality scores > 0.8',
            'realm_balancing': 'Optimize cross-realm coherence',
            'resonance_tuning': 'Enhance sacred geometry resonance patterns'
        }
    }
    
    try:
        with open('ubp_predictive_modeling_results.json', 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        # Create summary
        with open('ubp_predictive_modeling_summary.txt', 'w') as f:
            f.write("UBP PREDICTIVE MODELING SUMMARY\\n")
            f.write("="*50 + "\\n\\n")
            
            f.write("MODEL PERFORMANCE:\\n")
            for target_name, result in results.items():
                f.write(f"\\n{target_name}:\\n")
                if result['target_type'] == 'regression':
                    f.write(f"  R² Score: {result['metrics']['test_r2']:.3f}\\n")
                    f.write(f"  RMSE: {result['metrics']['test_rmse']:.3f}\\n")
                else:
                    f.write(f"  Accuracy: {result['metrics']['test_accuracy']:.3f}\\n")
                f.write(f"  Samples: {result['sample_count']}\\n")
            
            f.write("\\nUBP ANALYSIS:\\n")
            if 'nrci' in ubp_analysis:
                f.write(f"  NRCI Achievement: {ubp_analysis['nrci']['achievement_rate']:.1f}%\\n")
            if 'quality' in ubp_analysis:
                f.write(f"  High Quality Rate: {ubp_analysis['quality']['high_quality_rate']:.1f}%\\n")
            if 'realms' in ubp_analysis:
                f.write(f"  Dominant Realm: {max(ubp_analysis['realms'].items(), key=lambda x: x[1])[0]}\\n")
        
        print("  ✅ Saved results to ubp_predictive_modeling_results.json")
        print("  ✅ Saved summary to ubp_predictive_modeling_summary.txt")
        
    except Exception as e:
        print(f"  ❌ Error saving results: {e}")
    
    print("\\n" + "="*80)
    print("PHASE 6 COMPLETE")
    print("="*80)
    print("✅ UBP-enhanced predictive modeling complete")
    print("✅ Generated models for UBP feature prediction")
    print("✅ Created UBP-specific analysis and insights")
    print("✅ Ready for Phase 7: Interactive Periodic Neighborhood explorer")
    
    return final_results

if __name__ == "__main__":
    results = main()
