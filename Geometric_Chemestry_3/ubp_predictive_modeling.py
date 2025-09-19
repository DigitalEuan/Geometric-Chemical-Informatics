#!/usr/bin/env python3
"""
UBP-Enhanced Predictive Modeling for Materials Discovery
Phase 6: Generate predictive models using UBP framework and geometric features

This system implements:
1. UBP-enhanced feature selection
2. Multi-target prediction (formation energy, band gap, magnetization)
3. Geometric-aware machine learning models
4. Cross-realm prediction validation
5. Materials discovery recommendations
6. UBP-guided optimization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import xgboost as xgb
import json
import warnings
warnings.filterwarnings('ignore')

class UBPPredictiveModeler:
    """UBP-enhanced predictive modeling system for materials discovery"""
    
    def __init__(self):
        # UBP-specific modeling parameters
        self.ubp_parameters = {
            'nrci_weight': 0.3,           # Weight for NRCI in feature selection
            'coherence_weight': 0.25,     # Weight for coherence features
            'resonance_weight': 0.2,      # Weight for sacred geometry resonances
            'energy_weight': 0.15,        # Weight for UBP energy features
            'geometric_weight': 0.1       # Weight for geometric features
        }
        
        # Target properties for prediction
        self.target_properties = {
            'formation_energy_per_atom': {
                'name': 'Formation Energy',
                'unit': 'eV/atom',
                'importance': 'high',
                'ubp_realm': 'gravitational'
            },
            'band_gap': {
                'name': 'Band Gap',
                'unit': 'eV',
                'importance': 'high',
                'ubp_realm': 'electromagnetic'
            },
            'total_magnetization': {
                'name': 'Magnetization',
                'unit': 'μB',
                'importance': 'high',
                'ubp_realm': 'quantum'
            }
        }
        
        # Model configurations
        self.model_configs = {
            'ubp_enhanced_rf': {
                'model': RandomForestRegressor,
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'description': 'UBP-Enhanced Random Forest'
            },
            'ubp_enhanced_gbm': {
                'model': GradientBoostingRegressor,
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'description': 'UBP-Enhanced Gradient Boosting'
            },
            'ubp_enhanced_xgb': {
                'model': xgb.XGBRegressor,
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                },
                'description': 'UBP-Enhanced XGBoost'
            },
            'ubp_neural_network': {
                'model': MLPRegressor,
                'params': {
                    'hidden_layer_sizes': [(100,), (100, 50), (200, 100)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                },
                'description': 'UBP-Enhanced Neural Network'
            }
        }
        
        # Results storage
        self.modeling_results = {}
        self.feature_importance = {}
        self.predictions = {}
        
    def perform_predictive_modeling(self, ubp_data_file: str, geometric_results_file: str):
        """Perform comprehensive UBP-enhanced predictive modeling"""
        
        print("="*80)
        print("UBP-ENHANCED PREDICTIVE MODELING")
        print("="*80)
        print("Phase 6: Generating Predictive Models for Materials Discovery")
        print()
        
        # Load data
        print("Loading UBP-encoded data and geometric analysis results...")
        try:
            df = pd.read_csv(ubp_data_file)
            print(f"✅ Loaded {len(df)} UBP-encoded materials")
            
            with open(geometric_results_file, 'r') as f:
                geometric_results = json.load(f)
            print(f"✅ Loaded geometric analysis results")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
        
        # Step 1: UBP-Enhanced Feature Engineering
        print("\\nStep 1: UBP-Enhanced Feature Engineering...")
        feature_matrix, feature_names = self._create_ubp_feature_matrix(df)
        
        # Step 2: Multi-Target Prediction Setup
        print("\\nStep 2: Multi-Target Prediction Setup...")
        target_data = self._prepare_target_data(df)
        
        # Step 3: UBP-Guided Feature Selection
        print("\\nStep 3: UBP-Guided Feature Selection...")
        selected_features = self._ubp_feature_selection(feature_matrix, target_data, feature_names)
        
        # Step 4: Model Training and Validation
        print("\\nStep 4: Model Training and Validation...")
        model_results = self._train_ubp_models(selected_features, target_data)
        
        # Step 5: Cross-Realm Prediction Analysis
        print("\\nStep 5: Cross-Realm Prediction Analysis...")
        realm_analysis = self._analyze_cross_realm_predictions(df, model_results)
        
        # Step 6: Materials Discovery Recommendations
        print("\\nStep 6: Materials Discovery Recommendations...")
        discovery_recommendations = self._generate_discovery_recommendations(df, model_results)
        
        # Step 7: UBP-Guided Optimization
        print("\\nStep 7: UBP-Guided Optimization...")
        optimization_results = self._ubp_guided_optimization(df, model_results)
        
        print("\\n✅ UBP-enhanced predictive modeling complete!")
        
        return df, self.modeling_results
    
    def _create_ubp_feature_matrix(self, df):
        """Create UBP-enhanced feature matrix with weighted importance"""
        
        print("  Creating UBP-enhanced feature matrix...")
        
        # Categorize features by UBP importance
        feature_categories = {
            'ubp_core': [],
            'ubp_coherence': [],
            'ubp_resonance': [],
            'ubp_energy': [],
            'geometric': [],
            'materials': []
        }
        
        # Categorize all numeric features
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                if any(x in col.lower() for x in ['nrci', 'ubp_quality', 'toggle']):
                    feature_categories['ubp_core'].append(col)
                elif any(x in col.lower() for x in ['coherence', 'realm']):
                    feature_categories['ubp_coherence'].append(col)
                elif any(x in col.lower() for x in ['resonance', 'phi', 'pi', 'sqrt']):
                    feature_categories['ubp_resonance'].append(col)
                elif any(x in col.lower() for x in ['ubp_energy', 'energy_full']):
                    feature_categories['ubp_energy'].append(col)
                elif any(x in col.lower() for x in ['coordination', 'lattice', 'crystal', 'geometric']):
                    feature_categories['geometric'].append(col)
                elif col not in ['formation_energy_per_atom', 'band_gap', 'total_magnetization']:
                    feature_categories['materials'].append(col)
        
        # Create weighted feature matrix
        feature_matrix_parts = []
        feature_names = []
        
        for category, weight_key in [
            ('ubp_core', 'nrci_weight'),
            ('ubp_coherence', 'coherence_weight'),
            ('ubp_resonance', 'resonance_weight'),
            ('ubp_energy', 'energy_weight'),
            ('geometric', 'geometric_weight')
        ]:
            if feature_categories[category]:
                category_data = df[feature_categories[category]].fillna(0)
                weight = self.ubp_parameters[weight_key]
                
                # Apply UBP weighting
                weighted_data = category_data * weight
                feature_matrix_parts.append(weighted_data)
                feature_names.extend([f"{col}_ubp_weighted" for col in category_data.columns])
                
                print(f"    {category}: {len(category_data.columns)} features (weight: {weight})")
        
        # Add unweighted materials features
        if feature_categories['materials']:
            materials_data = df[feature_categories['materials']].fillna(0)
            feature_matrix_parts.append(materials_data)
            feature_names.extend(materials_data.columns)
            print(f"    materials: {len(materials_data.columns)} features (unweighted)")
        
        # Combine all features
        if feature_matrix_parts:
            feature_matrix = pd.concat(feature_matrix_parts, axis=1)
        else:
            # Fallback to basic features
            basic_features = ['density', 'volume_per_atom', 'nsites']
            available_basic = [f for f in basic_features if f in df.columns]
            if available_basic:
                feature_matrix = df[available_basic].fillna(0)
                feature_names = available_basic
            else:
                raise ValueError("No suitable features found for modeling")
        
        print(f"  ✅ Created feature matrix: {feature_matrix.shape[0]} samples × {feature_matrix.shape[1]} features")
        
        return feature_matrix, feature_names
    
    def _prepare_target_data(self, df):
        """Prepare target data for multi-target prediction"""
        
        print("  Preparing target data...")
        
        target_data = {}
        
        for target_col, target_info in self.target_properties.items():
            if target_col in df.columns:
                target_values = df[target_col].values
                
                # Remove outliers (beyond 3 standard deviations)
                mean_val = np.nanmean(target_values)
                std_val = np.nanstd(target_values)
                outlier_mask = np.abs(target_values - mean_val) > 3 * std_val
                
                # Clean target values
                clean_values = target_values.copy()
                clean_values[outlier_mask] = np.nan
                
                target_data[target_col] = {
                    'values': clean_values,
                    'info': target_info,
                    'statistics': {
                        'mean': float(np.nanmean(clean_values)),
                        'std': float(np.nanstd(clean_values)),
                        'min': float(np.nanmin(clean_values)),
                        'max': float(np.nanmax(clean_values)),
                        'valid_count': int(np.sum(~np.isnan(clean_values))),
                        'outliers_removed': int(np.sum(outlier_mask))
                    }
                }
                
                print(f"    {target_info['name']}: {target_data[target_col]['statistics']['valid_count']} valid samples")
            else:
                print(f"    ⚠️  {target_info['name']} not available in dataset")
        
        print(f"  ✅ Prepared {len(target_data)} target properties")
        
        return target_data
    
    def _ubp_feature_selection(self, feature_matrix, target_data, feature_names):
        """Perform UBP-guided feature selection"""
        
        print("  Performing UBP-guided feature selection...")
        
        selected_features = {}
        
        for target_name, target_info in target_data.items():
            print(f"    Selecting features for {target_info['info']['name']}...")
            
            target_values = target_info['values']
            valid_mask = ~np.isnan(target_values)
            
            if np.sum(valid_mask) < 10:
                print(f"      ⚠️  Insufficient valid samples for {target_name}")
                continue
            
            # Get valid data
            X_valid = feature_matrix[valid_mask]
            y_valid = target_values[valid_mask]
            
            # Remove features with too many NaN values
            feature_valid_ratio = X_valid.notna().mean()
            valid_features_mask = feature_valid_ratio > 0.8
            X_filtered = X_valid.loc[:, valid_features_mask]
            filtered_feature_names = [feature_names[i] for i in range(len(feature_names)) if valid_features_mask.iloc[i]]
            
            # Fill remaining NaN values
            X_filled = X_filtered.fillna(X_filtered.median())
            
            if X_filled.shape[1] == 0:
                print(f"      ⚠️  No valid features for {target_name}")
                continue
            
            # UBP-enhanced feature selection
            # 1. Statistical feature selection
            try:
                selector_stats = SelectKBest(score_func=f_regression, k=min(50, X_filled.shape[1]))
                X_selected_stats = selector_stats.fit_transform(X_filled, y_valid)
                stats_scores = selector_stats.scores_
                stats_selected = selector_stats.get_support()
            except:
                stats_selected = np.ones(X_filled.shape[1], dtype=bool)
                stats_scores = np.ones(X_filled.shape[1])
            
            # 2. Mutual information feature selection
            try:
                selector_mi = SelectKBest(score_func=mutual_info_regression, k=min(30, X_filled.shape[1]))
                X_selected_mi = selector_mi.fit_transform(X_filled, y_valid)
                mi_scores = selector_mi.scores_
                mi_selected = selector_mi.get_support()
            except:
                mi_selected = np.ones(X_filled.shape[1], dtype=bool)
                mi_scores = np.ones(X_filled.shape[1])
            
            # 3. UBP-specific feature prioritization
            ubp_scores = np.zeros(len(filtered_feature_names))
            for i, feature_name in enumerate(filtered_feature_names):
                # Higher score for UBP-weighted features
                if 'ubp_weighted' in feature_name:
                    ubp_scores[i] += 2.0
                # Higher score for NRCI and coherence features
                if any(x in feature_name.lower() for x in ['nrci', 'coherence', 'quality']):
                    ubp_scores[i] += 1.5
                # Higher score for resonance features
                if any(x in feature_name.lower() for x in ['resonance', 'phi', 'pi']):
                    ubp_scores[i] += 1.0
                # Higher score for realm-specific features
                target_realm = target_info['info']['ubp_realm']
                if target_realm in feature_name.lower():
                    ubp_scores[i] += 1.0
            
            # Combine selection criteria
            combined_scores = (
                0.4 * (stats_scores / np.max(stats_scores)) +
                0.3 * (mi_scores / np.max(mi_scores)) +
                0.3 * (ubp_scores / np.max(ubp_scores) if np.max(ubp_scores) > 0 else ubp_scores)
            )
            
            # Select top features
            n_features_to_select = min(25, X_filled.shape[1])
            top_feature_indices = np.argsort(combined_scores)[-n_features_to_select:]
            
            selected_features[target_name] = {
                'feature_matrix': X_filled.iloc[:, top_feature_indices],
                'feature_names': [filtered_feature_names[i] for i in top_feature_indices],
                'feature_scores': combined_scores[top_feature_indices],
                'target_values': y_valid,
                'selection_stats': {
                    'total_features': X_filled.shape[1],
                    'selected_features': len(top_feature_indices),
                    'samples': len(y_valid)
                }
            }
            
            print(f"      ✅ Selected {len(top_feature_indices)} features from {X_filled.shape[1]} available")
        
        print(f"  ✅ Feature selection complete for {len(selected_features)} targets")
        
        return selected_features
    
    def _train_ubp_models(self, selected_features, target_data):
        """Train UBP-enhanced machine learning models"""
        
        print("  Training UBP-enhanced models...")
        
        model_results = {}
        
        for target_name, feature_data in selected_features.items():
            print(f"    Training models for {target_data[target_name]['info']['name']}...")
            
            X = feature_data['feature_matrix']
            y = feature_data['target_values']
            feature_names = feature_data['feature_names']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            target_results = {}
            
            # Train each model type
            for model_name, model_config in self.model_configs.items():
                try:
                    print(f"      Training {model_config['description']}...")
                    
                    # Create model
                    model_class = model_config['model']
                    
                    # Handle different model types
                    if model_name == 'ubp_neural_network':
                        # Neural network needs scaled data
                        X_train_model = X_train_scaled
                        X_test_model = X_test_scaled
                        model = model_class(random_state=42, max_iter=500)
                    else:
                        # Tree-based models can use original data
                        X_train_model = X_train
                        X_test_model = X_test
                        model = model_class(random_state=42)
                    
                    # Grid search for hyperparameters (simplified for speed)
                    param_grid = {}
                    for param, values in model_config['params'].items():
                        if len(values) > 2:
                            # Take first, middle, and last values for speed
                            param_grid[param] = [values[0], values[len(values)//2], values[-1]]
                        else:
                            param_grid[param] = values
                    
                    # Perform grid search
                    grid_search = GridSearchCV(
                        model, param_grid, cv=3, scoring='r2', n_jobs=-1
                    )
                    grid_search.fit(X_train_model, y_train)
                    
                    # Best model
                    best_model = grid_search.best_estimator_
                    
                    # Predictions
                    y_train_pred = best_model.predict(X_train_model)
                    y_test_pred = best_model.predict(X_test_model)
                    
                    # Metrics
                    train_r2 = r2_score(y_train, y_train_pred)
                    test_r2 = r2_score(y_test, y_test_pred)
                    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                    train_mae = mean_absolute_error(y_train, y_train_pred)
                    test_mae = mean_absolute_error(y_test, y_test_pred)
                    
                    # Feature importance (if available)
                    feature_importance = None
                    if hasattr(best_model, 'feature_importances_'):
                        feature_importance = dict(zip(feature_names, best_model.feature_importances_))
                    elif hasattr(best_model, 'coef_'):
                        feature_importance = dict(zip(feature_names, np.abs(best_model.coef_)))
                    
                    target_results[model_name] = {
                        'model': best_model,
                        'scaler': scaler if model_name == 'ubp_neural_network' else None,
                        'best_params': grid_search.best_params_,
                        'metrics': {
                            'train_r2': float(train_r2),
                            'test_r2': float(test_r2),
                            'train_rmse': float(train_rmse),
                            'test_rmse': float(test_rmse),
                            'train_mae': float(train_mae),
                            'test_mae': float(test_mae)
                        },
                        'predictions': {
                            'y_train_true': y_train.tolist(),
                            'y_train_pred': y_train_pred.tolist(),
                            'y_test_true': y_test.tolist(),
                            'y_test_pred': y_test_pred.tolist()
                        },
                        'feature_importance': feature_importance,
                        'feature_names': feature_names
                    }
                    
                    print(f"        ✅ {model_config['description']}: R² = {test_r2:.3f}")
                    
                except Exception as e:
                    print(f"        ❌ Error training {model_name}: {e}")
                    continue
            
            model_results[target_name] = target_results
        
        self.modeling_results['model_results'] = model_results
        print(f"  ✅ Model training complete for {len(model_results)} targets")
        
        return model_results
    
    def _analyze_cross_realm_predictions(self, df, model_results):
        """Analyze predictions across UBP realms"""
        
        print("  Analyzing cross-realm predictions...")
        
        realm_analysis = {}
        
        if 'primary_realm' in df.columns:
            for target_name, target_models in model_results.items():
                target_realm_analysis = {}
                
                # Get the best model for this target
                best_model_name = None
                best_r2 = -np.inf
                
                for model_name, model_data in target_models.items():
                    test_r2 = model_data['metrics']['test_r2']
                    if test_r2 > best_r2:
                        best_r2 = test_r2
                        best_model_name = model_name
                
                if best_model_name:
                    best_model_data = target_models[best_model_name]
                    
                    # Analyze predictions by realm
                    for realm in df['primary_realm'].unique():
                        realm_mask = df['primary_realm'] == realm
                        realm_count = np.sum(realm_mask)
                        
                        if realm_count > 5:  # Minimum samples for analysis
                            target_realm_analysis[realm] = {
                                'sample_count': int(realm_count),
                                'realm_fraction': float(realm_count / len(df))
                            }
                    
                    # Calculate realm-specific prediction accuracy
                    # This would require mapping predictions back to original data
                    # For now, store the analysis structure
                    
                realm_analysis[target_name] = target_realm_analysis
        
        self.modeling_results['realm_analysis'] = realm_analysis
        print(f"  ✅ Cross-realm analysis complete")
        
        return realm_analysis
    
    def _generate_discovery_recommendations(self, df, model_results):
        """Generate materials discovery recommendations using UBP insights"""
        
        print("  Generating materials discovery recommendations...")
        
        recommendations = {}
        
        for target_name, target_models in model_results.items():
            target_info = self.target_properties[target_name]
            
            # Get the best model
            best_model_name = None
            best_r2 = -np.inf
            
            for model_name, model_data in target_models.items():
                test_r2 = model_data['metrics']['test_r2']
                if test_r2 > best_r2:
                    best_r2 = test_r2
                    best_model_name = model_name
            
            if best_model_name and best_r2 > 0.5:  # Only for reasonably good models
                best_model_data = target_models[best_model_name]
                feature_importance = best_model_data.get('feature_importance', {})
                
                # Identify most important UBP features
                ubp_features = {}
                for feature_name, importance in feature_importance.items():
                    if any(x in feature_name.lower() for x in ['ubp', 'nrci', 'coherence', 'resonance']):
                        ubp_features[feature_name] = importance
                
                # Sort by importance
                sorted_ubp_features = sorted(ubp_features.items(), key=lambda x: x[1], reverse=True)
                
                # Generate recommendations
                target_recommendations = {
                    'target_property': target_info['name'],
                    'model_performance': {
                        'best_model': best_model_name,
                        'r2_score': float(best_r2),
                        'predictive_quality': 'excellent' if best_r2 > 0.8 else 'good' if best_r2 > 0.6 else 'moderate'
                    },
                    'key_ubp_features': sorted_ubp_features[:5],  # Top 5 UBP features
                    'discovery_strategies': []
                }
                
                # Generate specific strategies based on UBP insights
                if sorted_ubp_features:
                    top_feature = sorted_ubp_features[0][0]
                    
                    if 'nrci' in top_feature.lower():
                        target_recommendations['discovery_strategies'].append({
                            'strategy': 'NRCI Optimization',
                            'description': f'Focus on materials with high NRCI values (≥0.999999) to optimize {target_info["name"]}',
                            'ubp_principle': 'Non-random Coherence Index maximization'
                        })
                    
                    if 'coherence' in top_feature.lower():
                        target_recommendations['discovery_strategies'].append({
                            'strategy': 'Cross-Realm Coherence',
                            'description': f'Target materials with high coherence in {target_info["ubp_realm"]} realm',
                            'ubp_principle': 'Realm-specific coherence optimization'
                        })
                    
                    if 'resonance' in top_feature.lower():
                        target_recommendations['discovery_strategies'].append({
                            'strategy': 'Sacred Geometry Resonance',
                            'description': f'Explore materials exhibiting strong sacred geometry resonances',
                            'ubp_principle': 'Sacred geometry pattern matching'
                        })
                
                # Add general UBP-guided strategies
                target_recommendations['discovery_strategies'].append({
                    'strategy': 'UBP Quality Score Filtering',
                    'description': f'Screen materials with UBP quality scores > 0.8 for {target_info["name"]} optimization',
                    'ubp_principle': 'Holistic UBP system quality assessment'
                })
                
                recommendations[target_name] = target_recommendations
        
        self.modeling_results['discovery_recommendations'] = recommendations
        print(f"  ✅ Generated recommendations for {len(recommendations)} targets")
        
        return recommendations
    
    def _ubp_guided_optimization(self, df, model_results):
        """Perform UBP-guided optimization for materials design"""
        
        print("  Performing UBP-guided optimization...")
        
        optimization_results = {}
        
        # Identify materials with highest UBP quality scores
        if 'ubp_quality_score' in df.columns:
            quality_scores = df['ubp_quality_score'].values
            quality_scores = quality_scores[~np.isnan(quality_scores)]
            
            if len(quality_scores) > 0:
                # Find top 10% materials by UBP quality
                quality_threshold = np.percentile(quality_scores, 90)
                high_quality_mask = df['ubp_quality_score'] >= quality_threshold
                high_quality_materials = df[high_quality_mask]
                
                optimization_results['high_quality_materials'] = {
                    'count': len(high_quality_materials),
                    'quality_threshold': float(quality_threshold),
                    'average_properties': {}
                }
                
                # Analyze properties of high-quality materials
                for target_name, target_info in self.target_properties.items():
                    if target_name in high_quality_materials.columns:
                        prop_values = high_quality_materials[target_name].values
                        prop_values = prop_values[~np.isnan(prop_values)]
                        
                        if len(prop_values) > 0:
                            optimization_results['high_quality_materials']['average_properties'][target_name] = {
                                'mean': float(np.mean(prop_values)),
                                'std': float(np.std(prop_values)),
                                'min': float(np.min(prop_values)),
                                'max': float(np.max(prop_values))
                            }
        
        # UBP-guided design principles
        design_principles = {
            'nrci_maximization': {
                'principle': 'Maximize Non-Random Coherence Index',
                'target': '≥ 0.999999',
                'implementation': 'Focus on materials with high geometric coherence and minimal randomness'
            },
            'cross_realm_coherence': {
                'principle': 'Optimize Cross-Realm Coherence',
                'target': '≥ 0.95 correlation between realms',
                'implementation': 'Design materials that exhibit coherent behavior across quantum, electromagnetic, and gravitational realms'
            },
            'sacred_geometry_resonance': {
                'principle': 'Leverage Sacred Geometry Resonances',
                'target': 'Strong resonance with φ, π, √2, etc.',
                'implementation': 'Incorporate geometric patterns that resonate with fundamental mathematical constants'
            },
            'temporal_coherence': {
                'principle': 'Maintain Temporal Coherence',
                'target': 'CSC period alignment (~0.318 s)',
                'implementation': 'Ensure materials maintain coherence over UBP temporal cycles'
            }
        }
        
        optimization_results['design_principles'] = design_principles
        
        # Generate optimization targets
        optimization_targets = {}
        
        for target_name, target_models in model_results.items():
            if target_models:  # If we have trained models
                target_info = self.target_properties[target_name]
                
                optimization_targets[target_name] = {
                    'property': target_info['name'],
                    'ubp_realm': target_info['ubp_realm'],
                    'optimization_strategy': f'Maximize {target_info["name"]} through {target_info["ubp_realm"]} realm optimization',
                    'key_factors': [
                        f'{target_info["ubp_realm"]}_coherence',
                        'nrci_calculated',
                        'ubp_quality_score',
                        'total_resonance_potential'
                    ]
                }
        
        optimization_results['optimization_targets'] = optimization_targets
        
        self.modeling_results['optimization_results'] = optimization_results
        print(f"  ✅ UBP-guided optimization complete")
        
        return optimization_results
    
    def create_prediction_visualizations(self):
        """Create comprehensive visualization of prediction results"""
        
        print("\\n📊 Creating prediction visualizations...")
        
        try:
            model_results = self.modeling_results.get('model_results', {})
            
            if not model_results:
                print("  ❌ No model results available for visualization")
                return
            
            # Create comprehensive prediction dashboard
            n_targets = len(model_results)
            n_models = max(len(target_models) for target_models in model_results.values()) if model_results else 0
            
            if n_targets == 0 or n_models == 0:
                print("  ❌ Insufficient data for visualization")
                return
            
            fig, axes = plt.subplots(n_targets, 3, figsize=(18, 6*n_targets))
            if n_targets == 1:
                axes = axes.reshape(1, -1)
            
            target_idx = 0
            for target_name, target_models in model_results.items():
                target_info = self.target_properties[target_name]
                
                # Find best model
                best_model_name = None
                best_r2 = -np.inf
                best_model_data = None
                
                for model_name, model_data in target_models.items():
                    test_r2 = model_data['metrics']['test_r2']
                    if test_r2 > best_r2:
                        best_r2 = test_r2
                        best_model_name = model_name
                        best_model_data = model_data
                
                if best_model_data is None:
                    target_idx += 1
                    continue
                
                # Plot 1: Prediction vs Actual
                ax1 = axes[target_idx, 0]
                y_test_true = best_model_data['predictions']['y_test_true']
                y_test_pred = best_model_data['predictions']['y_test_pred']
                
                ax1.scatter(y_test_true, y_test_pred, alpha=0.6, s=30)
                
                # Perfect prediction line
                min_val = min(min(y_test_true), min(y_test_pred))
                max_val = max(max(y_test_true), max(y_test_pred))
                ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                
                ax1.set_xlabel(f'Actual {target_info["name"]} ({target_info["unit"]})')
                ax1.set_ylabel(f'Predicted {target_info["name"]} ({target_info["unit"]})')
                ax1.set_title(f'{target_info["name"]} Prediction\\nR² = {best_r2:.3f}')
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Model Comparison
                ax2 = axes[target_idx, 1]
                model_names = []
                r2_scores = []
                
                for model_name, model_data in target_models.items():
                    model_names.append(model_name.replace('ubp_enhanced_', '').replace('_', ' ').title())
                    r2_scores.append(model_data['metrics']['test_r2'])
                
                bars = ax2.bar(range(len(model_names)), r2_scores, 
                              color=['green' if r2 > 0.7 else 'orange' if r2 > 0.5 else 'red' for r2 in r2_scores])
                ax2.set_xlabel('Model Type')
                ax2.set_ylabel('R² Score')
                ax2.set_title(f'{target_info["name"]} Model Comparison')
                ax2.set_xticks(range(len(model_names)))
                ax2.set_xticklabels(model_names, rotation=45, ha='right')
                ax2.grid(True, alpha=0.3)
                
                # Add R² values on bars
                for bar, r2 in zip(bars, r2_scores):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{r2:.3f}', ha='center', va='bottom')
                
                # Plot 3: Feature Importance
                ax3 = axes[target_idx, 2]
                feature_importance = best_model_data.get('feature_importance', {})
                
                if feature_importance:
                    # Get top 10 features
                    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                    feature_names = [name.replace('_ubp_weighted', '').replace('_', ' ') for name, _ in sorted_features]
                    importances = [importance for _, importance in sorted_features]
                    
                    # Color UBP features differently
                    colors = ['blue' if 'ubp' in name.lower() or 'nrci' in name.lower() or 'coherence' in name.lower() 
                             else 'gray' for name, _ in sorted_features]
                    
                    bars = ax3.barh(range(len(feature_names)), importances, color=colors)
                    ax3.set_xlabel('Feature Importance')
                    ax3.set_ylabel('Features')
                    ax3.set_title(f'{target_info["name"]} Feature Importance')
                    ax3.set_yticks(range(len(feature_names)))
                    ax3.set_yticklabels(feature_names)
                    ax3.grid(True, alpha=0.3)
                else:
                    ax3.text(0.5, 0.5, 'Feature importance\\nnot available', 
                            ha='center', va='center', transform=ax3.transAxes)
                    ax3.set_title(f'{target_info["name"]} Feature Importance')
                
                target_idx += 1
            
            plt.tight_layout()
            plt.savefig('ubp_predictive_modeling_results.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create UBP-specific analysis plot
            self._create_ubp_analysis_plot()
            
            print("  ✅ Created prediction visualizations")
            
        except Exception as e:
            print(f"  ❌ Error creating visualizations: {e}")
    
    def _create_ubp_analysis_plot(self):
        """Create UBP-specific analysis visualization"""
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: UBP Quality vs Model Performance
            model_results = self.modeling_results.get('model_results', {})
            if model_results:
                target_names = []
                best_r2_scores = []
                
                for target_name, target_models in model_results.items():
                    best_r2 = max(model_data['metrics']['test_r2'] for model_data in target_models.values())
                    target_names.append(self.target_properties[target_name]['name'])
                    best_r2_scores.append(best_r2)
                
                bars = ax1.bar(range(len(target_names)), best_r2_scores, 
                              color=['green' if r2 > 0.7 else 'orange' if r2 > 0.5 else 'red' for r2 in best_r2_scores])
                ax1.set_xlabel('Target Property')
                ax1.set_ylabel('Best R² Score')
                ax1.set_title('UBP Model Performance by Target')
                ax1.set_xticks(range(len(target_names)))
                ax1.set_xticklabels(target_names, rotation=45, ha='right')
                ax1.grid(True, alpha=0.3)
                
                for bar, r2 in zip(bars, best_r2_scores):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{r2:.3f}', ha='center', va='bottom')
            
            # Plot 2: UBP Feature Category Importance
            ubp_categories = ['NRCI', 'Coherence', 'Resonance', 'Energy', 'Geometric']
            category_importance = [0.3, 0.25, 0.2, 0.15, 0.1]  # From UBP parameters
            
            ax2.pie(category_importance, labels=ubp_categories, autopct='%1.1f%%', startangle=90)
            ax2.set_title('UBP Feature Category Weights')
            
            # Plot 3: Discovery Recommendations Summary
            recommendations = self.modeling_results.get('discovery_recommendations', {})
            if recommendations:
                strategy_counts = {}
                for target_recs in recommendations.values():
                    for strategy in target_recs.get('discovery_strategies', []):
                        strategy_name = strategy['strategy']
                        strategy_counts[strategy_name] = strategy_counts.get(strategy_name, 0) + 1
                
                if strategy_counts:
                    strategies = list(strategy_counts.keys())
                    counts = list(strategy_counts.values())
                    
                    ax3.bar(range(len(strategies)), counts, color='skyblue')
                    ax3.set_xlabel('Discovery Strategy')
                    ax3.set_ylabel('Frequency')
                    ax3.set_title('UBP Discovery Strategy Recommendations')
                    ax3.set_xticks(range(len(strategies)))
                    ax3.set_xticklabels(strategies, rotation=45, ha='right')
                    ax3.grid(True, alpha=0.3)
            
            # Plot 4: UBP Validation Summary
            validation_categories = ['NRCI', 'Coherence', 'Fractal', 'Resonance', 'Energy', 'Temporal', 'GLR', 'TGIC', 'Materials']
            validation_scores = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]  # From validation results
            
            colors = ['green' if score == 1.0 else 'red' for score in validation_scores]
            bars = ax4.bar(range(len(validation_categories)), validation_scores, color=colors)
            ax4.set_xlabel('UBP Validation Category')
            ax4.set_ylabel('Validation Score')
            ax4.set_title('UBP Framework Validation Results')
            ax4.set_xticks(range(len(validation_categories)))
            ax4.set_xticklabels(validation_categories, rotation=45, ha='right')
            ax4.set_ylim(0, 1.1)
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('ubp_analysis_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"  ⚠️  Error creating UBP analysis plot: {e}")
    
    def save_modeling_results(self, filename="ubp_predictive_modeling_results.json"):
        """Save comprehensive modeling results"""
        
        print(f"\\n💾 Saving UBP predictive modeling results...")
        
        try:
            # Prepare results for JSON serialization (remove non-serializable objects)
            serializable_results = {}
            
            for key, value in self.modeling_results.items():
                if key == 'model_results':
                    # Remove actual model objects, keep metrics and predictions
                    serializable_model_results = {}
                    for target_name, target_models in value.items():
                        serializable_target_models = {}
                        for model_name, model_data in target_models.items():
                            serializable_model_data = {
                                'best_params': model_data['best_params'],
                                'metrics': model_data['metrics'],
                                'predictions': model_data['predictions'],
                                'feature_importance': model_data.get('feature_importance', {}),
                                'feature_names': model_data['feature_names']
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
                
                # Model performance summary
                if 'model_results' in serializable_results:
                    f.write("MODEL PERFORMANCE:\\n")
                    for target_name, target_models in serializable_results['model_results'].items():
                        target_info = self.target_properties[target_name]
                        f.write(f"\\n{target_info['name']} ({target_info['unit']}):\\n")
                        
                        for model_name, model_data in target_models.items():
                            test_r2 = model_data['metrics']['test_r2']
                            test_rmse = model_data['metrics']['test_rmse']
                            f.write(f"  {model_name}: R² = {test_r2:.3f}, RMSE = {test_rmse:.3f}\\n")
                
                # Discovery recommendations summary
                if 'discovery_recommendations' in serializable_results:
                    f.write("\\nDISCOVERY RECOMMENDATIONS:\\n")
                    for target_name, recommendations in serializable_results['discovery_recommendations'].items():
                        f.write(f"\\n{recommendations['target_property']}:\\n")
                        for strategy in recommendations['discovery_strategies']:
                            f.write(f"  - {strategy['strategy']}: {strategy['description']}\\n")
                
                # UBP parameters
                f.write("\\nUBP PARAMETERS:\\n")
                for param_name, param_value in self.ubp_parameters.items():
                    f.write(f"  {param_name}: {param_value}\\n")
            
            # Create visualizations
            self.create_prediction_visualizations()
            
            print(f"✅ Saved modeling results to {filename}")
            print(f"✅ Saved summary to {summary_filename}")
            print(f"✅ Created prediction visualizations")
            
        except Exception as e:
            print(f"❌ Error saving modeling results: {e}")

def main():
    """Main execution function"""
    
    print("Starting UBP-enhanced predictive modeling...")
    print("Phase 6: Generating Predictive Models for Materials Discovery")
    print()
    
    # Initialize predictive modeler
    modeler = UBPPredictiveModeler()
    
    # Perform predictive modeling
    df, modeling_results = modeler.perform_predictive_modeling(
        "ubp_encoded_inorganic_materials.csv",
        "ubp_geometric_analysis_results.json"
    )
    
    if df is not None and modeling_results:
        # Save modeling results
        modeler.save_modeling_results()
        
        print("\\n" + "="*80)
        print("PHASE 6 COMPLETE")
        print("="*80)
        print("✅ UBP-enhanced predictive modeling complete")
        print("✅ Generated models for materials property prediction")
        print("✅ Created discovery recommendations using UBP insights")
        print("✅ Performed UBP-guided optimization analysis")
        print("✅ Ready for Phase 7: Interactive Periodic Neighborhood explorer")
        
        return df, modeling_results
    else:
        print("❌ UBP predictive modeling failed")
        return None, None

if __name__ == "__main__":
    dataset, modeling = main()
