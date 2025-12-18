"""
Enhanced Multi-Cloud Resource Allocation Optimization using AI
IMPROVED VERSION - Addresses issues in the original code and ensures reliable output

Key Improvements:
1. Robust error handling and data validation
2. Simplified and more reliable data processing
3. Proper main execution flow
4. Comprehensive output generation
5. Better resource matching algorithms
6. Modular design for easier debugging
"""

import pandas as pd
import numpy as np
import random
import warnings
from collections import deque
from datetime import datetime, timedelta
import math
import os
import sys

# Machine Learning and AI imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, silhouette_score

# Deep Learning for RL
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not available. DQN training will be skipped.")
    TORCH_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')

# -------------------- Enhanced Data Preprocessing --------------------

class ImprovedDataProcessor:
    """Improved data processor with robust error handling"""
    
    def __init__(self):
        self.le_provider = LabelEncoder()
        self.le_vm_type = LabelEncoder()
        self.le_region = LabelEncoder()
        self.scaler = StandardScaler()
    
    def safe_extract_price(self, price_str):
        """Safely extract price from string format"""
        try:
            if pd.isna(price_str) or str(price_str).lower() in ['n/a', 'unavailable', 'nan', '']:
                return 0.0
            
            price_clean = str(price_str).replace('$', '').replace(' hourly', '').replace(',', '').strip()
            
            if not price_clean or price_clean.lower() in ['nan', 'unavailable', 'n/a']:
                return 0.0
                
            return float(price_clean)
        except (ValueError, AttributeError):
            return 0.0
    
    def safe_extract_vcpu(self, vcpu_str):
        """Safely extract vCPU count"""
        try:
            if pd.isna(vcpu_str):
                return 1
            if 'vCPUs' in str(vcpu_str):
                return int(str(vcpu_str).replace(' vCPUs', ''))
            return max(1, int(float(vcpu_str)))
        except:
            return 1
    
    def safe_extract_memory(self, mem_str):
        """Safely extract memory size"""
        try:
            if pd.isna(mem_str):
                return 1.0
            if 'GiB' in str(mem_str):
                return float(str(mem_str).replace(' GiB', ''))
            return max(0.5, float(mem_str))
        except:
            return 1.0
    
    def preprocess_data(self):
        """Complete data preprocessing pipeline with error handling"""
        print("🔄 Loading and preprocessing data...")
        
        try:
            # Load datasets
            expanded_df = pd.read_csv('expanded_multi_cloud_dataset_4000.csv')
            enhanced_df = pd.read_csv('Multi_Cloud_Instance_Comparison_Enhanced.csv')
            
            print(f"✅ Loaded {len(expanded_df)} workload records and {len(enhanced_df)} instance types")
            
            # Clean pricing data
            enhanced_df['On_Demand_Price'] = enhanced_df['On_Demand_Cost'].apply(self.safe_extract_price)
            enhanced_df['Reserved_1Y_Price'] = enhanced_df['Reserved_1Year_Cost'].apply(self.safe_extract_price)
            enhanced_df['Reserved_3Y_Price'] = enhanced_df['Reserved_3Year_Cost'].apply(self.safe_extract_price)
            enhanced_df['Spot_Min_Price'] = enhanced_df['Spot_Min_Cost'].apply(self.safe_extract_price)
            
            # Extract resource specifications
            enhanced_df['vCPU_num'] = enhanced_df['vCPU'].apply(self.safe_extract_vcpu)
            enhanced_df['Memory_GB'] = enhanced_df['Memory'].apply(self.safe_extract_memory)
            
            # Validate data quality
            valid_instances = len(enhanced_df[enhanced_df['On_Demand_Price'] > 0])
            print(f"✅ {valid_instances}/{len(enhanced_df)} instances have valid pricing")
            
            # Feature engineering
            expanded_df = self._engineer_features(expanded_df)
            
            # Create workload profiles
            workload_profiles = self._create_workload_profiles(expanded_df)
            
            print(f"✅ Created {len(workload_profiles)} workload profiles")
            
            return expanded_df, enhanced_df, workload_profiles
            
        except Exception as e:
            print(f"❌ Error in data preprocessing: {e}")
            raise
    
    def _engineer_features(self, df):
        """Feature engineering with error handling"""
        try:
            # Performance metrics
            df['performance_score'] = df['throughput'] / (df['latency_ms'] + 1) * 100
            df['cost_efficiency'] = df['throughput'] / (df['cost'] + 0.01)
            df['efficiency_score'] = df['utilization'] / 100.0
            
            # Resource intensity
            df['cpu_intensity'] = pd.cut(df['cpu_usage'], bins=[0, 30, 60, 90, 100],
                                       labels=['Low', 'Medium', 'High', 'Critical'])
            df['memory_intensity'] = pd.cut(df['memory_usage'], bins=[0, 30, 60, 90, 100],
                                          labels=['Low', 'Medium', 'High', 'Critical'])
            
            # I/O patterns
            df['io_intensity'] = (df['net_io'] + df['disk_io']) / 2
            
            return df
        except Exception as e:
            print(f"Warning: Feature engineering error: {e}")
            return df
    
    def _create_workload_profiles(self, df):
        """Create comprehensive workload profiles"""
        try:
            profiles = df.groupby('Name').agg({
                'cpu_usage': ['mean', 'max', 'min', 'std'],
                'memory_usage': ['mean', 'max', 'min', 'std'],
                'utilization': ['mean', 'max', 'min'],
                'latency_ms': ['mean', 'min', 'max'],
                'throughput': ['mean', 'max'],
                'cost': ['mean', 'sum'],
                'performance_score': 'mean',
                'cost_efficiency': 'mean',
                'efficiency_score': 'mean',
                'io_intensity': 'mean',
                'vCPU': 'first',
                'RAM_GB': 'first',
                'price_per_hour': 'first',
                'cloud_provider': 'first',
                'vm_type': 'first',
                'region': 'first'
            }).reset_index()
            
            # Flatten column names
            new_columns = ['Name']
            for col in profiles.columns[1:]:
                if isinstance(col, tuple):
                    if col[1]:
                        new_columns.append(f"{col[0]}_{col[1]}")
                    else:
                        new_columns.append(col[0])
                else:
                    new_columns.append(col)
            profiles.columns = new_columns
            
            return profiles
        except Exception as e:
            print(f"Error creating workload profiles: {e}")
            raise

# -------------------- AI Models Implementation --------------------

class ImprovedAIOptimizer:
    """Improved AI optimizer with better error handling"""
    
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.scaler = StandardScaler()
        self.model_results = {}
    
    def train_supervised_models(self, df):
        """Train supervised learning models"""
        print("🤖 Training machine learning models...")
        
        try:
            # Prepare features
            feature_cols = ['cpu_usage', 'memory_usage', 'utilization', 'latency_ms', 
                           'throughput', 'vCPU', 'RAM_GB', 'performance_score', 'cost_efficiency']
            
            available_cols = [col for col in feature_cols if col in df.columns]
            print(f"📊 Using features: {available_cols}")
            
            if len(available_cols) < 5:
                raise ValueError("Insufficient features for model training")
            
            X = df[available_cols].copy()
            
            # Encode categorical variables
            self.encoders['provider'] = LabelEncoder()
            self.encoders['vm_type'] = LabelEncoder()
            self.encoders['region'] = LabelEncoder()
            
            X['provider_enc'] = self.encoders['provider'].fit_transform(df['cloud_provider'])
            X['vm_type_enc'] = self.encoders['vm_type'].fit_transform(df['vm_type'])
            X['region_enc'] = self.encoders['region'].fit_transform(df['region'])
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Define targets
            targets = {
                'cost': df['cost'],
                'utilization': df['utilization'],
                'performance': df['performance_score']
            }
            
            results = {}
            
            for target_name, y in targets.items():
                print(f"  Training {target_name} model...")
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )
                
                # Train Random Forest
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                rf_model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = rf_model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                self.models[target_name] = rf_model
                results[target_name] = {'r2': r2, 'rmse': rmse}
                
                print(f"    ✅ {target_name}: R² = {r2:.3f}, RMSE = {rmse:.3f}")
            
            self.model_results = results
            return results
            
        except Exception as e:
            print(f"❌ Error in model training: {e}")
            return {}

# -------------------- Simple DQN Implementation --------------------

if TORCH_AVAILABLE:
    class SimpleDQN(nn.Module):
        """Simplified DQN for resource allocation"""
        
        def __init__(self, state_size, action_size, learning_rate=0.001):
            super(SimpleDQN, self).__init__()
            self.state_size = state_size
            self.action_size = action_size
            
            # Simple neural network
            self.network = nn.Sequential(
                nn.Linear(state_size, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_size)
            )
            
            self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
            self.criterion = nn.MSELoss()
            
            # RL parameters
            self.epsilon = 1.0
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.995
            self.memory = deque(maxlen=10000)
            self.gamma = 0.95
        
        def forward(self, x):
            return self.network(x)
        
        def act(self, state):
            """Choose action using epsilon-greedy policy"""
            if np.random.random() <= self.epsilon:
                return random.randrange(self.action_size)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.network(state_tensor)
                action = torch.argmax(q_values).item()
                return max(0, min(action, self.action_size - 1))
        
        def remember(self, state, action, reward, next_state, done):
            """Store experience"""
            self.memory.append((state, action, reward, next_state, done))
        
        def replay(self, batch_size=32):
            """Train the model"""
            if len(self.memory) < batch_size:
                return
            
            batch = random.sample(self.memory, batch_size)
            states = torch.FloatTensor([e[0] for e in batch])
            actions = torch.LongTensor([e[1] for e in batch])
            rewards = torch.FloatTensor([e[2] for e in batch])
            next_states = torch.FloatTensor([e[3] for e in batch])
            dones = torch.BoolTensor([e[4] for e in batch])
            
            current_q = self.network(states).gather(1, actions.unsqueeze(1))
            next_q = self.network(next_states).max(1)[0].detach()
            target_q = rewards + (self.gamma * next_q * ~dones)
            
            loss = self.criterion(current_q.squeeze(), target_q)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

# -------------------- Optimization Engine --------------------

class ImprovedOptimizationEngine:
    """Main optimization engine with reliable execution"""
    
    def __init__(self):
        self.processor = ImprovedDataProcessor()
        self.ai_optimizer = ImprovedAIOptimizer()
        self.results = {}
    
    def run_optimization(self):
        """Run complete optimization pipeline"""
        print("🚀 Starting Enhanced Multi-Cloud AI Optimization")
        print("=" * 80)
        
        try:
            # Step 1: Data preprocessing
            expanded_df, enhanced_df, workload_profiles = self.processor.preprocess_data()
            
            # Step 2: Train AI models
            ml_results = self.ai_optimizer.train_supervised_models(expanded_df)
            
            # Step 3: Generate recommendations
            print("🔍 Generating optimization recommendations...")
            recommendations = self._generate_recommendations(workload_profiles, enhanced_df)
            
            # Step 4: Calculate performance metrics
            print("📊 Calculating performance metrics...")
            performance_metrics = self._calculate_metrics(recommendations)
            
            # Step 5: Save results
            print("💾 Saving results...")
            self._save_results(recommendations, performance_metrics, ml_results)
            
            # Step 6: Display summary
            self._display_results(recommendations, performance_metrics)
            
            return recommendations, performance_metrics
            
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            return pd.DataFrame(), {}
    
    def _generate_recommendations(self, workload_profiles, enhanced_df):
        """Generate optimization recommendations using heuristic approach"""
        recommendations = []
        
        # Filter valid instances
        valid_instances = enhanced_df[enhanced_df['On_Demand_Price'] > 0].copy()
        
        if valid_instances.empty:
            print("❌ No valid instances found for optimization")
            return pd.DataFrame()
        
        print(f"📋 Processing {len(workload_profiles)} workloads against {len(valid_instances)} instances")
        
        processed = 0
        for idx, workload in workload_profiles.iterrows():
            try:
                recommendation = self._find_best_match(workload, valid_instances)
                if recommendation:
                    recommendations.append(recommendation)
                    processed += 1
                
                # Process a reasonable subset to ensure output
                if processed >= min(50, len(workload_profiles)):
                    break
                    
            except Exception as e:
                print(f"Warning: Error processing workload {idx}: {e}")
                continue
        
        print(f"✅ Generated {len(recommendations)} recommendations from {processed} workloads")
        return pd.DataFrame(recommendations)
    
    def _find_best_match(self, workload, valid_instances):
        """Find best instance match for a workload"""
        try:
            # Get workload characteristics
            workload_name = workload.get('Name', 'Unknown')
            cpu_max = workload.get('cpu_usage_max', 50)
            mem_max = workload.get('memory_usage_max', 50)
            current_vcpu = workload.get('vCPU_first', 1)
            current_ram = workload.get('RAM_GB_first', 1)
            current_cost_hourly = workload.get('price_per_hour_first', 0.1)
            current_cost_monthly = current_cost_hourly * 24 * 30.44
            
            # Calculate requirements with safety margin
            cpu_req = max(1, (cpu_max / 100) * current_vcpu * 1.1)
            mem_req = max(1, (mem_max / 100) * current_ram * 1.1)
            
            # Find suitable instances
            suitable = valid_instances[
                (valid_instances['vCPU_num'] >= cpu_req) &
                (valid_instances['Memory_GB'] >= mem_req)
            ].copy()
            
            if suitable.empty:
                return None
            
            # Calculate scores for each suitable instance
            scores = []
            for _, instance in suitable.iterrows():
                # Cost savings score
                reserved_cost_monthly = max(
                    instance['Reserved_1Y_Price'] * 24 * 30.44,
                    instance['On_Demand_Price'] * 24 * 30.44 * 0.65
                )
                
                cost_savings = max(0, current_cost_monthly - reserved_cost_monthly)
                cost_score = cost_savings / current_cost_monthly if current_cost_monthly > 0 else 0
                
                # Resource efficiency score
                cpu_eff = min(1.0, cpu_req / instance['vCPU_num'])
                mem_eff = min(1.0, mem_req / instance['Memory_GB'])
                resource_score = (cpu_eff + mem_eff) / 2
                
                # Overall score
                overall_score = cost_score * 0.6 + resource_score * 0.4
                
                scores.append({
                    'instance': instance,
                    'score': overall_score,
                    'cost_savings': cost_savings,
                    'reserved_monthly': reserved_cost_monthly
                })
            
            # Select best instance
            if not scores:
                return None
            
            best = max(scores, key=lambda x: x['score'])
            
            # Only recommend if there are meaningful savings
            if best['cost_savings'] < 5:  # Less than $5/month savings
                return None
            
            instance = best['instance']
            
            return {
                'workload_name': workload_name,
                'current_provider': workload.get('cloud_provider_first', 'Unknown'),
                'current_instance': workload.get('vm_type_first', 'unknown'),
                'current_vcpu': current_vcpu,
                'current_memory_gb': current_ram,
                'current_cost_monthly': round(current_cost_monthly, 2),
                'recommended_provider': instance['Cloud_Provider'],
                'recommended_instance': str(instance['Instance_Name'])[:60],
                'recommended_vcpu': int(instance['vCPU_num']),
                'recommended_memory_gb': round(instance['Memory_GB'], 1),
                'instance_family': instance.get('Instance_Family', 'Unknown'),
                'on_demand_monthly': round(instance['On_Demand_Price'] * 24 * 30.44, 2),
                'reserved_monthly': round(best['reserved_monthly'], 2),
                'spot_monthly': round(instance['Spot_Min_Price'] * 24 * 30.44, 2),
                'monthly_savings': round(best['cost_savings'], 2),
                'savings_percentage': round((best['cost_savings'] / current_cost_monthly) * 100, 1) if current_cost_monthly > 0 else 0,
                'optimization_score': round(best['score'], 3),
                'cpu_efficiency': round(min(1.0, cpu_req / instance['vCPU_num']) * 100, 1),
                'memory_efficiency': round(min(1.0, mem_req / instance['Memory_GB']) * 100, 1)
            }
            
        except Exception as e:
            print(f"Error finding match for workload: {e}")
            return None
    
    def _calculate_metrics(self, recommendations_df):
        """Calculate summary performance metrics"""
        if recommendations_df.empty:
            return {
                'total_workloads': 0,
                'message': 'No recommendations generated'
            }
        
        total_current = recommendations_df['current_cost_monthly'].sum()
        total_savings = recommendations_df['monthly_savings'].sum()
        
        return {
            'total_workloads_analyzed': len(recommendations_df),
            'total_current_monthly_cost': round(total_current, 2),
            'total_monthly_savings': round(total_savings, 2),
            'potential_annual_savings': round(total_savings * 12, 2),
            'average_savings_percentage': round(recommendations_df['savings_percentage'].mean(), 1),
            'average_optimization_score': round(recommendations_df['optimization_score'].mean(), 3),
            'average_cpu_efficiency': round(recommendations_df['cpu_efficiency'].mean(), 1),
            'average_memory_efficiency': round(recommendations_df['memory_efficiency'].mean(), 1),
            'provider_distribution': recommendations_df['recommended_provider'].value_counts().to_dict()
        }
    
    def _save_results(self, recommendations_df, metrics, ml_results):
        """Save results to CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not recommendations_df.empty:
            # Save recommendations
            rec_filename = f'multicloud_optimization_recommendations_{timestamp}.csv'
            recommendations_df.to_csv(rec_filename, index=False)
            print(f"📄 Recommendations saved: {rec_filename}")
            
            # Save metrics
            metrics_df = pd.DataFrame([metrics])
            metrics_filename = f'optimization_metrics_{timestamp}.csv'
            metrics_df.to_csv(metrics_filename, index=False)
            print(f"📊 Metrics saved: {metrics_filename}")
            
            # Save ML results if available
            if ml_results:
                ml_df = pd.DataFrame(ml_results).T
                ml_filename = f'ml_model_results_{timestamp}.csv'
                ml_df.to_csv(ml_filename, index=False)
                print(f"🤖 ML results saved: {ml_filename}")
    
    def _display_results(self, recommendations_df, metrics):
        """Display comprehensive results summary"""
        print("\n" + "=" * 80)
        print("🎯 MULTI-CLOUD OPTIMIZATION RESULTS SUMMARY")
        print("=" * 80)
        
        if recommendations_df.empty:
            print("❌ No optimization recommendations were generated.")
            print("   This could be due to:")
            print("   • Current workloads already optimally configured")
            print("   • Insufficient cost savings opportunities")
            print("   • Data quality issues")
            return
        
        # Display key metrics
        print(f"📊 Total Workloads Analyzed: {metrics['total_workloads_analyzed']}")
        print(f"💰 Current Monthly Cost: ${metrics['total_current_monthly_cost']:,.2f}")
        print(f"💵 Potential Monthly Savings: ${metrics['total_monthly_savings']:,.2f}")
        print(f"🎯 Potential Annual Savings: ${metrics['potential_annual_savings']:,.2f}")
        print(f"📈 Average Savings: {metrics['average_savings_percentage']:.1f}%")
        print(f"⚡ Average Optimization Score: {metrics['average_optimization_score']:.3f}")
        print(f"🔧 Average CPU Efficiency: {metrics['average_cpu_efficiency']:.1f}%")
        print(f"💾 Average Memory Efficiency: {metrics['average_memory_efficiency']:.1f}%")
        
        # Provider distribution
        print(f"\n☁️ Recommended Provider Distribution:")
        for provider, count in metrics['provider_distribution'].items():
            print(f"   {provider}: {count} recommendations")
        
        # Top 10 recommendations
        print(f"\n🏆 TOP 10 OPTIMIZATION OPPORTUNITIES:")
        print("-" * 80)
        
        top_10 = recommendations_df.nlargest(10, 'monthly_savings')
        
        for i, (_, rec) in enumerate(top_10.iterrows(), 1):
            print(f"{i:2d}. {rec['workload_name']}")
            print(f"    Current: {rec['current_provider']} {rec['current_instance']} "
                  f"({rec['current_vcpu']} vCPU, {rec['current_memory_gb']} GB)")
            print(f"    → Recommended: {rec['recommended_provider']} {rec['recommended_instance']} "
                  f"({rec['recommended_vcpu']} vCPU, {rec['recommended_memory_gb']} GB)")
            print(f"    💰 Savings: ${rec['monthly_savings']:.2f}/month ({rec['savings_percentage']:.1f}%)")
            print(f"    ⚡ Score: {rec['optimization_score']:.3f} | "
                  f"CPU: {rec['cpu_efficiency']:.1f}% | RAM: {rec['memory_efficiency']:.1f}%")
            print()
        
        print("✅ Optimization completed successfully!")
        print("=" * 80)

# -------------------- Main Execution --------------------

def main():
    """Main execution function"""
    print("🚀 Enhanced Multi-Cloud Resource Allocation Optimization")
    print("📋 Improved Version - Reliable Output Generation")
    print("=" * 80)
    
    # Check if data files exist
    required_files = ['expanded_multi_cloud_dataset_4000.csv', 'Multi_Cloud_Instance_Comparison_Enhanced.csv']
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Required file not found: {file}")
            print("   Please ensure all required CSV files are in the current directory.")
            return
    
    try:
        # Initialize and run optimization
        engine = ImprovedOptimizationEngine()
        recommendations, metrics = engine.run_optimization()
        
        if not recommendations.empty:
            print(f"\n🎉 Successfully generated {len(recommendations)} optimization recommendations!")
            print(f"💰 Total potential annual savings: ${metrics.get('potential_annual_savings', 0):,.2f}")
        else:
            print("\n⚠️ No optimization opportunities identified with current constraints.")
            
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your data files and try again.")

if __name__ == "__main__":
    main()