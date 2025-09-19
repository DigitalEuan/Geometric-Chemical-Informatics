#!/usr/bin/env python3
"""
UBP-Enhanced Predictive Modeling for Materials Discovery (Adapted)
Phase 6: Generate predictive models using available UBP features

This system implements:
1. UBP quality score prediction
2. Realm classification prediction
3. NRCI optimization prediction
4. Cross-realm coherence prediction
5. Materials discovery recommendations based on UBP principles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
import json
import warnings
warnings.filterwarnings('ignore')

class UBPAdaptedPredictiveModeler:
    """UBP-enhanced predictive modeling system adapted for available features"""
    
    def __init__(self):
        # UBP prediction targets based on available data
        self.prediction_targets = {
            'ubp_quality_score': {
                'type': 'regression',
                'name': 'UBP Quality Score',
                'description': 'Overall UBP system quality assessment',
                'importance': 'critical'
            },
            'nrci_calculated': {
                'type': 'regression',
                'name': 'NRCI Value',
                'description': 'Non-Random Coherence Index prediction',
                'importance': 'critical'
            },
            'primary_realm': {
                'type': 'classification',
                'name': 'Primary Realm',
                'description': 'UBP realm classification',
                'importance': 'high'
            },
            'system_coherence': {
                'type': 'regression',
                'name': 'System Coherence',
                'description': 'Overall system coherence prediction',
                'importance': 'high'
            },
            'total_resonance_potential': {
                'type': 'regression',
                'name': 'Resonance Potential',
                'description': 'Sacred geometry resonance potential',
                'importance': 'medium'
            }
        }
        
        # Model configurations
        self.model_configs = {
            'regression': {
                'random_forest': {
                    'model': RandomForestRegressor,
                    'params': {
                        'n_estimators': [100, 200],
                        'max_depth': [10, 20, None],
                        'min_samples_split': [2, 5]
                    }
                },
                'gradient_boosting': {
                    'model': GradientBoostingRegressor,
                    'params': {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.05, 0.1],
                        'max_depth': [3, 5]
                    }
                },
                'neural_network': {
                    'model': MLPRegressor,
                    'params': {
                        'hidden_layer_sizes': [(100,), (100, 50)],
                        'alpha': [0.001, 0.01]
                    }
                }
            },
            'classification': {
                'random_forest': {
                    'model': RandomForestClassifier,
                    'params': {
                        'n_estimators': [100, 200],
                        'max_depth': [10, 20],
                        'min_samples_split': [2, 5]
                    }
                },
                'neural_network': {
                    'model': MLPClassifier,
                    'params': {
                        'hidden_layer_sizes': [(100,), (100, 50)],
                        'alpha': [0.001, 0.01]
                    }
                }
            }
        }
        
        # Results storage
        self.modeling_results = {}
        
    def perform_ubp_predictive_modeling(self, ubp_data_file: str):
        """Perform UBP-enhanced predictive modeling with available features"""
        
        print("="*80)
        print("UBP-ENHANCED PREDICTIVE MODELING (ADAPTED)")
        print("="*80)
        print("Phase 6: Generating Predictive Models for UBP Features")
        print()
        
        # Load data
        print("Loading UBP-encoded data...")
        try:
            df = pd.read_csv(ubp_data_file)
            print(f"✅ Loaded {len(df)} UBP-encoded materials with {len(df.columns)} features")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
        
        # Step 1: Feature Engineering for UBP Predictions
        print("\\nStep 1: UBP Feature Engineering...")
        feature_matrix, feature_names = self._create_ubp_prediction_features(df)
        
        # Step 2: Target Preparation
        print("\\nStep 2: Target Preparation...")
        target_data = self._prepare_ubp_targets(df)
        
        # Step 3: Model Training for Each Target
        print("\\nStep 3: UBP Model Training...")
        model_results = self._train_ubp_prediction_models(feature_matrix, target_data, feature_names)
        
        # Step 4: UBP-Specific Analysis
        print("\\nStep 4: UBP-Specific Analysis...")
        ubp_analysis = self._perform_ubp_analysis(df, model_results)
        
        # Step 5: Materials Discovery Insights
        print("\\nStep 5: Materials Discovery Insights...")
        discovery_insights = self._generate_ubp_discovery_insights(df, model_results)
        
        # Step 6: UBP Optimization Recommendations
        print("\\nStep 6: UBP Optimization Recommendations...")
        optimization_recommendations = self._generate_ubp_optimization_recommendations(df, model_results)
        
        print("\\n✅ UBP-enhanced predictive modeling complete!")
        
        return df, self.modeling_results
    
    def _create_ubp_prediction_features(self, df):
        """Create feature matrix for UBP predictions"""
        
        print("  Creating UBP prediction features...")
        
        # Define feature categories
        feature_categories = {
            'realm_encodings': [col for col in df.columns if any(x in col for x in ['quantum_', 'electromagnetic_', 'gravitational_', 'biological_', 'cosmological_', 'nuclear_', 'optical_']) and col != 'primary_realm'],
            'resonance_features': [col for col in df.columns if 'resonance' in col and col != 'total_resonance_potential'],
            'coherence_features': [col for col in df.columns if 'coherence' in col],
            'toggle_features': [col for col in df.columns if 'toggle' in col],
            'interaction_features': [col for col in df.columns if 'interaction' in col],
            'energy_features': [col for col in df.columns if 'energy' in col and col != 'ubp_energy_full']
        }
        
        # Combine all features
        all_features = []
        for category, features in feature_categories.items():
            all_features.extend(features)
            print(f"    {category}: {len(features)} features")
        
        # Remove target columns from features
        target_columns = list(self.prediction_targets.keys())
        feature_columns = [col for col in all_features if col not in target_columns and col in df.columns]
        
        # Create feature matrix
        if feature_columns:
            feature_matrix = df[feature_columns].copy()
        else:
            print("    ⚠️  No valid feature columns found, using basic numeric columns")
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            excluded_targets = list(self.prediction_targets.keys())
            feature_columns = [col for col in numeric_columns if col not in excluded_targets]
            feature_matrix = df[feature_columns].copy()
        
        # Handle non-numeric columns
        for col in feature_matrix.columns:
            if feature_matrix[col].dtype == 'object':
                # Try to convert to numeric, if fails, encode categorically
                try:
                    feature_matrix[col] = pd.to_numeric(feature_matrix[col], errors='coerce')
                except:
                    le = LabelEncoder()
                    feature_matrix[col] = le.fit_transform(feature_matrix[col].astype(str))
        
        # Fill NaN values
        feature_matrix = feature_matrix.fillna(feature_matrix.median())
        
        print(f"  ✅ Created feature matrix: {feature_matrix.shape[0]} samples × {feature_matrix.shape[1]} features")
        
        return feature_matrix, feature_columns
    
    def _prepare_ubp_targets(self, df):
        """Prepare target variables for UBP predictions"""
        
        print("  Preparing UBP targets...")
        
        target_data = {}
        
        for target_name, target_info in self.prediction_targets.items():
            if target_name in df.columns:
                target_values = df[target_name].copy()
                
                # Handle different target types
                if target_info['type'] == 'classification':
                    if target_values.dtype == 'object':
                        le = LabelEncoder()
                        target_values = le.fit_transform(target_values.astype(str))
                        target_info['label_encoder'] = le
                        target_info['classes'] = le.classes_
                    
                    valid_mask = ~pd.isna(target_values)
                    
                elif target_info['type'] == 'regression':
                    # Remove outliers for regression targets
                    if target_values.dtype in ['int64', 'float64']:
                        Q1 = target_values.quantile(0.25)
                        Q3 = target_values.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        outlier_mask = (target_values < lower_bound) | (target_values > upper_bound)
                        target_values[outlier_mask] = np.nan
                    
                    valid_mask = ~pd.isna(target_values)
                
                if np.sum(valid_mask) > 10:  # Minimum samples required
                    target_data[target_name] = {
                        'values': target_values,
                        'valid_mask': valid_mask,
                        'info': target_info,
                        'statistics': {
                            'valid_count': int(np.sum(valid_mask)),
                            'total_count': len(target_values)
                        }
                    }
                    
                    if target_info['type'] == 'regression':
                        valid_values = target_values[valid_mask]
                        target_data[target_name]['statistics'].update({
                            'mean': float(np.mean(valid_values)),
                            'std': float(np.std(valid_values)),
                            'min': float(np.min(valid_values)),
                            'max': float(np.max(valid_values))
                        })
                    
                    print(f"    {target_info['name']}: {target_data[target_name]['statistics']['valid_count']} valid samples")
                else:
                    print(f"    ⚠️  {target_info['name']}: insufficient valid samples ({np.sum(valid_mask)})")
            else:
                print(f"    ⚠️  {target_info['name']}: not available in dataset")
        
        print(f"  ✅ Prepared {len(target_data)} UBP targets")
        
        return target_data
    
    def _train_ubp_prediction_models(self, feature_matrix, target_data, feature_names):
        """Train prediction models for UBP targets"""
        
        print("  Training UBP prediction models...")
        
        model_results = {}
        
        for target_name, target_info in target_data.items():
            print(f"    Training models for {target_info['info']['name']}...")
            
            target_values = target_info['values']
            valid_mask = target_info['valid_mask']
            target_type = target_info['info']['type']
            
            # Get valid data
            X_valid = feature_matrix[valid_mask]
            y_valid = target_values[valid_mask]
            
            if len(y_valid) < 20:
                print(f"      ⚠️  Insufficient data for {target_name}")
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_valid, y_valid, test_size=0.2, random_state=42, stratify=y_valid if target_type == 'classification' else None
            )
            
            # Scale features for neural networks
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            target_results = {}
            
            # Train models based on target type
            model_configs = self.model_configs[target_type]
            
            for model_name, model_config in model_configs.items():
                try:
                    print(f"      Training {model_name}...")
                    
                    # Choose appropriate data
                    if model_name == 'neural_network':
                        X_train_model = X_train_scaled
                        X_test_model = X_test_scaled
                    else:
                        X_train_model = X_train
                        X_test_model = X_test
                    
                    # Create and train model
                    model_class = model_config['model']
                    model = model_class(random_state=42)
                    
                    # Grid search
                    grid_search = GridSearchCV(
                        model, model_config['params'], cv=3, 
                        scoring='r2' if target_type == 'regression' else 'accuracy',
                        n_jobs=-1
                    )
                    grid_search.fit(X_train_model, y_train)
                    
                    # Best model
                    best_model = grid_search.best_estimator_
                    
                    # Predictions
                    y_train_pred = best_model.predict(X_train_model)
                    y_test_pred = best_model.predict(X_test_model)
                    
                    # Metrics
                    if target_type == 'regression':
                        train_score = r2_score(y_train, y_train_pred)
                        test_score = r2_score(y_test, y_test_pred)
                        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                        
                        metrics = {
                            'train_r2': float(train_score),
                            'test_r2': float(test_score),
                            'train_rmse': float(train_rmse),
                            'test_rmse': float(test_rmse)
                        }
                    else:
                        train_score = accuracy_score(y_train, y_train_pred)
                        test_score = accuracy_score(y_test, y_test_pred)
                        
                        metrics = {
                            'train_accuracy': float(train_score),
                            'test_accuracy': float(test_score)
                        }
                    
                    # Feature importance
                    feature_importance = None
                    if hasattr(best_model, 'feature_importances_'):
                        feature_importance = dict(zip(feature_names, best_model.feature_importances_))
                    elif hasattr(best_model, 'coef_'):
                        if target_type == 'classification' and len(best_model.coef_.shape) > 1:
                            # Multi-class classification
                            feature_importance = dict(zip(feature_names, np.mean(np.abs(best_model.coef_), axis=0)))
                        else:
                            feature_importance = dict(zip(feature_names, np.abs(best_model.coef_.flatten())))
                    
                    target_results[model_name] = {
                        'model': best_model,
                        'scaler': scaler if model_name == 'neural_network' else None,
                        'best_params': grid_search.best_params_,
                        'metrics': metrics,
                        'predictions': {
                            'y_train_true': y_train.tolist() if hasattr(y_train, 'tolist') else list(y_train),
                            'y_train_pred': y_train_pred.tolist() if hasattr(y_train_pred, 'tolist') else list(y_train_pred),
                            'y_test_true': y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test),
                            'y_test_pred': y_test_pred.tolist() if hasattr(y_test_pred, 'tolist') else list(y_test_pred)
                        },
                        'feature_importance': feature_importance
                    }
                    
                    score_name = 'R²' if target_type == 'regression' else 'Accuracy'
                    print(f"        ✅ {model_name}: {score_name} = {test_score:.3f}")
                    
                except Exception as e:
                    print(f"        ❌ Error training {model_name}: {e}")
                    continue
            
            if target_results:
                model_results[target_name] = target_results
        
        self.modeling_results['model_results'] = model_results
        print(f"  ✅ Model training complete for {len(model_results)} targets")
        
        return model_results
    
    def _perform_ubp_analysis(self, df, model_results):
        """Perform UBP-specific analysis"""
        
        print("  Performing UBP-specific analysis...")
        
        ubp_analysis = {}
        
        # Analyze NRCI distribution and prediction quality
        if 'nrci_calculated' in model_results:
            nrci_models = model_results['nrci_calculated']
            best_nrci_model = max(nrci_models.items(), key=lambda x: x[1]['metrics'].get('test_r2', 0))
            
            ubp_analysis['nrci_prediction'] = {
                'best_model': best_nrci_model[0],
                'performance': best_nrci_model[1]['metrics'],
                'nrci_target_achievement': {
                    'target': 0.999999,
                    'current_achievement': float(np.sum(df['nrci_calculated'] >= 0.999999) / len(df) * 100),
                    'prediction_quality': 'excellent' if best_nrci_model[1]['metrics'].get('test_r2', 0) > 0.8 else 'good'
                }
            }
        
        # Analyze realm distribution and classification
        if 'primary_realm' in model_results:
            realm_models = model_results['primary_realm']
            best_realm_model = max(realm_models.items(), key=lambda x: x[1]['metrics'].get('test_accuracy', 0))
            
            realm_distribution = df['primary_realm'].value_counts().to_dict()
            
            ubp_analysis['realm_classification'] = {
                'best_model': best_realm_model[0],
                'performance': best_realm_model[1]['metrics'],
                'realm_distribution': realm_distribution,
                'dominant_realm': max(realm_distribution.items(), key=lambda x: x[1])[0]
            }
        
        # Analyze UBP quality score prediction
        if 'ubp_quality_score' in model_results:
            quality_models = model_results['ubp_quality_score']
            best_quality_model = max(quality_models.items(), key=lambda x: x[1]['metrics'].get('test_r2', 0))
            
            quality_scores = df['ubp_quality_score'].dropna()
            
            ubp_analysis['quality_prediction'] = {
                'best_model': best_quality_model[0],
                'performance': best_quality_model[1]['metrics'],
                'quality_distribution': {
                    'mean': float(quality_scores.mean()),
                    'std': float(quality_scores.std()),
                    'high_quality_percentage': float(np.sum(quality_scores > 0.8) / len(quality_scores) * 100)
                }
            }
        
        # Analyze coherence patterns
        coherence_columns = [col for col in df.columns if 'coherence' in col]
        if coherence_columns:
            coherence_data = df[coherence_columns].dropna()
            
            ubp_analysis['coherence_patterns'] = {
                'coherence_features': coherence_columns,
                'mean_coherence': {col: float(coherence_data[col].mean()) for col in coherence_columns},
                'coherence_correlations': coherence_data.corr().to_dict()
            }
        
        self.modeling_results['ubp_analysis'] = ubp_analysis
        print(f"  ✅ UBP analysis complete")
        
        return ubp_analysis
    
    def _generate_ubp_discovery_insights(self, df, model_results):
        """Generate materials discovery insights based on UBP predictions"""
        
        print("  Generating UBP discovery insights...")
        
        discovery_insights = {}
        
        # High-quality materials identification
        if 'ubp_quality_score' in df.columns:
            quality_scores = df['ubp_quality_score'].dropna()
            high_quality_threshold = quality_scores.quantile(0.9)
            high_quality_materials = df[df['ubp_quality_score'] >= high_quality_threshold]
            
            discovery_insights['high_quality_materials'] = {
                'threshold': float(high_quality_threshold),
                'count': len(high_quality_materials),
                'percentage': float(len(high_quality_materials) / len(df) * 100),
                'characteristics': {}
            }
            
            # Analyze characteristics of high-quality materials
            if 'primary_realm' in high_quality_materials.columns:
                realm_dist = high_quality_materials['primary_realm'].value_counts().to_dict()
                discovery_insights['high_quality_materials']['characteristics']['realm_preference'] = realm_dist
            
            if 'nrci_calculated' in high_quality_materials.columns:
                nrci_stats = {
                    'mean': float(high_quality_materials['nrci_calculated'].mean()),
                    'min': float(high_quality_materials['nrci_calculated'].min()),
                    'max': float(high_quality_materials['nrci_calculated'].max())
                }
                discovery_insights['high_quality_materials']['characteristics']['nrci_stats'] = nrci_stats
        
        # NRCI optimization insights
        if 'nrci_calculated' in df.columns:
            nrci_values = df['nrci_calculated'].dropna()
            nrci_target = 0.999999
            
            high_nrci_materials = df[df['nrci_calculated'] >= nrci_target]
            
            discovery_insights['nrci_optimization'] = {
                'target': nrci_target,
                'achievement_rate': float(len(high_nrci_materials) / len(df) * 100),
                'optimization_strategies': []
            }
            
            # Identify features correlated with high NRCI
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            nrci_correlations = df[numeric_columns].corrwith(df['nrci_calculated']).abs().sort_values(ascending=False)
            
            top_nrci_features = nrci_correlations.head(5).to_dict()
            discovery_insights['nrci_optimization']['key_features'] = top_nrci_features
            
            # Generate optimization strategies
            for feature, correlation in list(top_nrci_features.items())[:3]:
                if correlation > 0.3:
                    strategy = f"Optimize {feature.replace('_', ' ')} (correlation: {correlation:.3f})"
                    discovery_insights['nrci_optimization']['optimization_strategies'].append(strategy)
        
        # Realm-specific insights
        if 'primary_realm' in df.columns:
            realm_insights = {}
            
            for realm in df['primary_realm'].unique():
                realm_materials = df[df['primary_realm'] == realm]
                
                realm_insights[realm] = {
                    'count': len(realm_materials),
                    'percentage': float(len(realm_materials) / len(df) * 100)
                }
                
                # Average UBP metrics for this realm
                if 'ubp_quality_score' in realm_materials.columns:
                    realm_insights[realm]['avg_quality'] = float(realm_materials['ubp_quality_score'].mean())
                
                if 'nrci_calculated' in realm_materials.columns:
                    realm_insights[realm]['avg_nrci'] = float(realm_materials['nrci_calculated'].mean())
                
                if 'total_resonance_potential' in realm_materials.columns:
                    realm_insights[realm]['avg_resonance'] = float(realm_materials['total_resonance_potential'].mean())
            
            discovery_insights['realm_insights'] = realm_insights
        
        self.modeling_results['discovery_insights'] = discovery_insights
        print(f"  ✅ Discovery insights generated")
        
        return discovery_insights
    
    def _generate_ubp_optimization_recommendations(self, df, model_results):
        """Generate UBP-based optimization recommendations"""
        
        print("  Generating UBP optimization recommendations...")
        
        recommendations = {
            'nrci_optimization': {
                'target': 'Achieve NRCI ≥ 0.999999 for maximum coherence',
                'strategies': [
                    'Focus on materials with high quantum realm coherence',
                    'Optimize toggle patterns for minimal randomness',
                    'Enhance cross-realm coherence interactions',
                    'Implement TGIC constraints (3-6-9 structure)'
                ]
            },
            'quality_enhancement': {
                'target': 'Maximize UBP quality score > 0.8',
                'strategies': [
                    'Balance coherence across all UBP realms',
                    'Optimize sacred geometry resonance patterns',
                    'Enhance temporal coherence synchronization',
                    'Minimize system entropy and randomness'
                ]
            },
            'realm_optimization': {
                'target': 'Achieve optimal realm distribution and coherence',
                'strategies': []
            },
            'discovery_priorities': {
                'high_priority': [
                    'Materials with NRCI > 0.999999',
                    'Compounds showing strong phi (φ) resonance',
                    'Systems with high cross-realm coherence',
                    'Materials in quantum-electromagnetic realm overlap'
                ],
                'medium_priority': [
                    'Compounds with balanced resonance patterns',
                    'Materials showing temporal coherence',
                    'Systems with optimized toggle patterns'
                ],
                'research_directions': [
                    'Investigate fractal dimension optimization',
                    'Explore CARFE field equation applications',
                    'Develop UBP-guided synthesis protocols',
                    'Create real-time NRCI monitoring systems'
                ]
            }
        }
        
        # Add model-specific recommendations
        if model_results:
            best_performing_targets = []
            
            for target_name, target_models in model_results.items():
                best_score = 0
                best_model = None
                
                for model_name, model_data in target_models.items():
                    score = model_data['metrics'].get('test_r2', model_data['metrics'].get('test_accuracy', 0))
                    if score > best_score:
                        best_score = score
                        best_model = model_name
                
                if best_score > 0.7:  # Good prediction quality
                    best_performing_targets.append({
                        'target': target_name,
                        'score': best_score,
                        'model': best_model
                    })
            
            if best_performing_targets:
                recommendations['predictive_modeling'] = {
                    'reliable_predictions': best_performing_targets,
                    'recommendation': 'Use these high-quality models for materials screening and optimization'
                }
        
        # Add realm-specific strategies
        if 'primary_realm' in df.columns:
            realm_distribution = df['primary_realm'].value_counts()
            dominant_realm = realm_distribution.index[0]
            
            realm_strategies = {
                'quantum': 'Focus on magnetic properties and electron spin optimization',
                'electromagnetic': 'Optimize electronic band structure and conductivity',
                'gravitational': 'Enhance structural stability and mechanical properties',
                'biological': 'Explore biocompatibility and organic interfaces',
                'cosmological': 'Investigate large-scale ordering and symmetry',
                'nuclear': 'Focus on nuclear stability and isotopic properties',
                'optical': 'Optimize photonic and optical properties'
            }
            
            if dominant_realm in realm_strategies:
                recommendations['realm_optimization']['strategies'].append(
                    f"Primary focus: {realm_strategies[dominant_realm]} (dominant realm: {dominant_realm})"
                )
        
        self.modeling_results['optimization_recommendations'] = recommendations
        print(f"  ✅ Optimization recommendations generated")
        
        return recommendations
    
    def create_ubp_visualizations(self):
        """Create comprehensive UBP prediction visualizations"""
        
        print("\\n📊 Creating UBP prediction visualizations...")
        
        try:
            model_results = self.modeling_results.get('model_results', {})
            
            if not model_results:
                print("  ❌ No model results available for visualization")
                return
            
            # Create main prediction dashboard
            n_targets = len(model_results)
            if n_targets == 0:
                print("  ❌ No targets to visualize")
                return
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            plot_idx = 0
            
            # Plot 1: Model Performance Summary
            target_names = []
            best_scores = []
            score_types = []
            
            for target_name, target_models in model_results.items():
                target_info = self.prediction_targets[target_name]
                target_names.append(target_info['name'])
                
                # Find best model
                best_score = 0
                for model_name, model_data in target_models.items():
                    if target_info['type'] == 'regression':
                        score = model_data['metrics'].get('test_r2', 0)
                        score_type = 'R²'
                    else:
                        score = model_data['metrics'].get('test_accuracy', 0)
                        score_type = 'Accuracy'
                    
                    if score > best_score:
                        best_score = score
                
                best_scores.append(best_score)
                score_types.append(score_type)
            
            if target_names and best_scores:
                colors = ['green' if score > 0.7 else 'orange' if score > 0.5 else 'red' for score in best_scores]
                bars = axes[plot_idx].bar(range(len(target_names)), best_scores, color=colors)
                axes[plot_idx].set_xlabel('UBP Target')
                axes[plot_idx].set_ylabel('Best Model Score')
                axes[plot_idx].set_title('UBP Prediction Model Performance')
                axes[plot_idx].set_xticks(range(len(target_names)))
                axes[plot_idx].set_xticklabels(target_names, rotation=45, ha='right')
                axes[plot_idx].grid(True, alpha=0.3)
                
                # Add score labels
                for bar, score in zip(bars, best_scores):
                    height = bar.get_height()
                    axes[plot_idx].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                       f'{score:.3f}', ha='center', va='bottom')
                
                plot_idx += 1
            
            # Plot 2: NRCI Distribution (if available)
            if 'nrci_calculated' in model_results:
                nrci_models = model_results['nrci_calculated']
                best_nrci_model = max(nrci_models.items(), key=lambda x: x[1]['metrics'].get('test_r2', 0))
                
                y_test_true = best_nrci_model[1]['predictions']['y_test_true']
                y_test_pred = best_nrci_model[1]['predictions']['y_test_pred']
                
                axes[plot_idx].scatter(y_test_true, y_test_pred, alpha=0.6, s=30)
                
                # Perfect prediction line
                min_val = min(min(y_test_true), min(y_test_pred))
                max_val = max(max(y_test_true), max(y_test_pred))
                axes[plot_idx].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                
                axes[plot_idx].set_xlabel('Actual NRCI')
                axes[plot_idx].set_ylabel('Predicted NRCI')
                axes[plot_idx].set_title(f'NRCI Prediction\\nR² = {best_nrci_model[1]["metrics"]["test_r2"]:.3f}')
                axes[plot_idx].grid(True, alpha=0.3)
                
                plot_idx += 1
            
            # Plot 3: UBP Quality Score Distribution (if available)
            ubp_analysis = self.modeling_results.get('ubp_analysis', {})
            if 'quality_prediction' in ubp_analysis:
                quality_dist = ubp_analysis['quality_prediction']['quality_distribution']
                
                # Create histogram
                axes[plot_idx].hist([quality_dist['mean']], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                axes[plot_idx].axvline(0.8, color='red', linestyle='--', label='High Quality Threshold')
                axes[plot_idx].set_xlabel('UBP Quality Score')
                axes[plot_idx].set_ylabel('Frequency')
                axes[plot_idx].set_title('UBP Quality Score Distribution')
                axes[plot_idx].legend()
                axes[plot_idx].grid(True, alpha=0.3)
                
                plot_idx += 1
            
            # Plot 4: Realm Distribution (if available)
            if 'realm_classification' in ubp_analysis:
                realm_dist = ubp_analysis['realm_classification']['realm_distribution']
                
                realms = list(realm_dist.keys())
                counts = list(realm_dist.values())
                
                axes[plot_idx].pie(counts, labels=realms, autopct='%1.1f%%', startangle=90)
                axes[plot_idx].set_title('UBP Realm Distribution')
                
                plot_idx += 1
            
            # Plot 5: Feature Importance (for best model)
            if model_results:
                # Find the best performing model overall
                best_overall_model = None
                best_overall_score = 0
                best_target_name = None
                
                for target_name, target_models in model_results.items():
                    for model_name, model_data in target_models.items():
                        score = model_data['metrics'].get('test_r2', model_data['metrics'].get('test_accuracy', 0))
                        if score > best_overall_score:
                            best_overall_score = score
                            best_overall_model = model_data
                            best_target_name = target_name
                
                if best_overall_model and best_overall_model.get('feature_importance'):
                    feature_importance = best_overall_model['feature_importance']
                    
                    # Get top 10 features
                    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                    feature_names = [name.replace('_', ' ') for name, _ in sorted_features]
                    importances = [importance for _, importance in sorted_features]
                    
                    axes[plot_idx].barh(range(len(feature_names)), importances, color='lightcoral')
                    axes[plot_idx].set_xlabel('Feature Importance')
                    axes[plot_idx].set_ylabel('Features')
                    axes[plot_idx].set_title(f'Top Features for {self.prediction_targets[best_target_name]["name"]}')
                    axes[plot_idx].set_yticks(range(len(feature_names)))
                    axes[plot_idx].set_yticklabels(feature_names)
                    axes[plot_idx].grid(True, alpha=0.3)
                    
                    plot_idx += 1
            
            # Plot 6: UBP Optimization Recommendations
            recommendations = self.modeling_results.get('optimization_recommendations', {})
            if 'discovery_priorities' in recommendations:
                priorities = recommendations['discovery_priorities']
                
                categories = ['High Priority', 'Medium Priority', 'Research Directions']
                counts = [
                    len(priorities.get('high_priority', [])),
                    len(priorities.get('medium_priority', [])),
                    len(priorities.get('research_directions', []))
                ]
                
                axes[plot_idx].bar(categories, counts, color=['red', 'orange', 'blue'])
                axes[plot_idx].set_xlabel('Priority Category')
                axes[plot_idx].set_ylabel('Number of Items')
                axes[plot_idx].set_title('UBP Discovery Priorities')
                axes[plot_idx].grid(True, alpha=0.3)
                
                plot_idx += 1
            
            # Hide unused subplots
            for i in range(plot_idx, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig('ubp_predictive_modeling_results.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("  ✅ Created UBP prediction visualizations")
            
        except Exception as e:
            print(f"  ❌ Error creating visualizations: {e}")
    
    def save_ubp_modeling_results(self, filename="ubp_predictive_modeling_results.json"):
        """Save UBP modeling results"""
        
        print(f"\\n💾 Saving UBP modeling results...")
        
        try:
            # Prepare results for JSON serialization
            serializable_results = {}
            
            for key, value in self.modeling_results.items():
                if key == 'model_results':
                    # Remove model objects, keep metrics and predictions
                    serializable_model_results = {}
                    for target_name, target_models in value.items():
                        serializable_target_models = {}
                        for model_name, model_data in target_models.items():
                            serializable_model_data = {
                                'best_params': model_data['best_params'],
                                'metrics': model_data['metrics'],
                                'predictions': model_data['predictions'],
                                'feature_importance': model_data.get('feature_importance', {})
                            }
                            serializable_target_models[model_name] = serializable_model_data
                        serializable_model_results[target_name] = serializable_target_models
                    serializable_results[key] = serializable_model_results
                else:
                    serializable_results[key] = value
            
            # Save main results
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            # Create summary report
            summary_filename = filename.replace('.json', '_summary.txt')
            with open(summary_filename, 'w') as f:
                f.write("UBP PREDICTIVE MODELING SUMMARY\\n")
                f.write("="*50 + "\\n\\n")
                
                # Model performance
                if 'model_results' in serializable_results:
                    f.write("MODEL PERFORMANCE:\\n")
                    for target_name, target_models in serializable_results['model_results'].items():
                        target_info = self.prediction_targets[target_name]
                        f.write(f"\\n{target_info['name']} ({target_info['type']}):\\n")
                        
                        for model_name, model_data in target_models.items():
                            if target_info['type'] == 'regression':
                                score = model_data['metrics']['test_r2']
                                f.write(f"  {model_name}: R² = {score:.3f}\\n")
                            else:
                                score = model_data['metrics']['test_accuracy']
                                f.write(f"  {model_name}: Accuracy = {score:.3f}\\n")
                
                # UBP analysis
                if 'ubp_analysis' in serializable_results:
                    f.write("\\nUBP ANALYSIS:\\n")
                    ubp_analysis = serializable_results['ubp_analysis']
                    
                    if 'nrci_prediction' in ubp_analysis:
                        nrci_info = ubp_analysis['nrci_prediction']
                        f.write(f"  NRCI Achievement: {nrci_info['nrci_target_achievement']['current_achievement']:.1f}%\\n")
                    
                    if 'realm_classification' in ubp_analysis:
                        realm_info = ubp_analysis['realm_classification']
                        f.write(f"  Dominant Realm: {realm_info['dominant_realm']}\\n")
                
                # Discovery insights
                if 'discovery_insights' in serializable_results:
                    f.write("\\nDISCOVERY INSIGHTS:\\n")
                    insights = serializable_results['discovery_insights']
                    
                    if 'high_quality_materials' in insights:
                        hq_info = insights['high_quality_materials']
                        f.write(f"  High-Quality Materials: {hq_info['count']} ({hq_info['percentage']:.1f}%)\\n")
                    
                    if 'nrci_optimization' in insights:
                        nrci_info = insights['nrci_optimization']
                        f.write(f"  NRCI Target Achievement: {nrci_info['achievement_rate']:.1f}%\\n")
            
            # Create visualizations
            self.create_ubp_visualizations()
            
            print(f"✅ Saved UBP modeling results to {filename}")
            print(f"✅ Saved summary to {summary_filename}")
            print(f"✅ Created UBP visualizations")
            
        except Exception as e:
            print(f"❌ Error saving UBP modeling results: {e}")

def main():
    """Main execution function"""
    
    print("Starting UBP-enhanced predictive modeling (adapted)...")
    print("Phase 6: Generating Predictive Models for UBP Features")
    print()
    
    # Initialize adapted modeler
    modeler = UBPAdaptedPredictiveModeler()
    
    # Perform UBP predictive modeling
    df, modeling_results = modeler.perform_ubp_predictive_modeling(
        "ubp_encoded_inorganic_materials.csv"
    )
    
    if df is not None and modeling_results:
        # Save modeling results
        modeler.save_ubp_modeling_results()
        
        print("\\n" + "="*80)
        print("PHASE 6 COMPLETE")
        print("="*80)
        print("✅ UBP-enhanced predictive modeling complete")
        print("✅ Generated models for UBP feature prediction")
        print("✅ Created UBP-specific discovery insights")
        print("✅ Generated optimization recommendations")
        print("✅ Ready for Phase 7: Interactive Periodic Neighborhood explorer")
        
        return df, modeling_results
    else:
        print("❌ UBP predictive modeling failed")
        return None, None

if __name__ == "__main__":
    dataset, modeling = main()
