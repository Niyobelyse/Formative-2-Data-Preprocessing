"""
Product Recommendation Module
Predicts product recommendations for authenticated users
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ProductRecommendationModule:
    """
    Product Recommendation System
    Generates personalized product predictions
    """
    
    def __init__(self, model_dir='models', data_dir='data/processed'):
        """
        Initialize product recommendation module
        
        Args:
            model_dir: Directory containing trained models
            data_dir: Directory containing processed data
        """
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        
        print("Loading Product Recommendation Model...")
        
        try:
            # Load trained model
            self.model = joblib.load(self.model_dir / 'product_recommendation_model.pkl')
            self.scaler = joblib.load(self.model_dir / 'scaler_product.pkl')
            self.label_encoder = joblib.load(self.model_dir / 'label_encoder_product.pkl')
            
            # Try to load feature columns if available
            try:
                self.feature_columns = joblib.load(self.model_dir / 'feature_columns_product.pkl')
            except:
                self.feature_columns = None
                print("   ⚠️  Feature columns not found, will use all numeric columns")
            
            # Load merged dataset for customer profiles
            self.merged_data = pd.read_csv(self.data_dir / 'merged_dataset.csv')
            
            print("✓ Product Recommendation Model loaded successfully\n")
            
        except FileNotFoundError as e:
            print(f"❌ Error: Files not found")
            print(f"   Missing: {e.filename}")
            raise
    
    def get_customer_profile(self, customer_id=None, member_id=None):
        """
        Get customer profile from merged dataset
        
        Args:
            customer_id: Specific customer ID
            member_id: Member ID (alternative to customer_id)
            
        Returns:
            Customer feature dictionary
        """
        try:
            if customer_id:
                customer_data = self.merged_data[self.merged_data['customer_id'] == customer_id].iloc[0]
            elif member_id:
                # Use member_id to get a sample customer profile
                # In production, you'd map member_id to customer_id
                customer_data = self.merged_data.sample(1).iloc[0]
            else:
                # Random customer for demo
                customer_data = self.merged_data.sample(1).iloc[0]
            
            # Remove non-feature columns
            exclude_cols = ['customer_id', 'transaction_id', 'transaction_date']
            if 'product_purchased' in customer_data.index:
                exclude_cols.append('product_purchased')
            
            feature_dict = {col: customer_data[col] for col in customer_data.index 
                          if col not in exclude_cols}
            
            return feature_dict
            
        except Exception as e:
            print(f"⚠️  Could not retrieve customer profile: {str(e)}")
            return self._get_default_profile()
    
    def _get_default_profile(self):
        """Generate a default customer profile"""
        return {
            'followers': 5000,
            'engagement_rate': 3.5,
            'total_spent': 1500,
            'transaction_count': 10,
            'avg_transaction_value': 150,
            'recency': 30,
            'frequency': 10,
            'monetary': 1500
        }
    
    def predict(self, customer_features):
        """
        Predict product recommendation
        
        Args:
            customer_features: Dictionary or DataFrame of customer features
            
        Returns:
            tuple: (product_name, confidence, top_3_products)
        """
        print(f"🔍 Generating product recommendation...")
        
        try:
            # Convert to DataFrame if dict
            if isinstance(customer_features, dict):
                customer_df = pd.DataFrame([customer_features])
            else:
                customer_df = customer_features.copy()
            
            # Ensure all required columns are present
            if self.feature_columns is not None:
                for col in self.feature_columns:
                    if col not in customer_df.columns:
                        customer_df[col] = 0
                customer_df = customer_df[self.feature_columns]
            
            # Fill any missing values
            customer_df = customer_df.fillna(0)
            
            # Scale features
            features_scaled = self.scaler.transform(customer_df)
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Get product name
            product_name = self.label_encoder.inverse_transform([prediction])[0]
            confidence = probabilities[prediction]
            
            # Get top 3 products
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            top_3_products = [
                (self.label_encoder.inverse_transform([idx])[0], probabilities[idx])
                for idx in top_3_indices
            ]
            
            return product_name, confidence, top_3_products
            
        except Exception as e:
            print(f"❌ Error making prediction: {str(e)}")
            return "Unknown Product", 0.0, []


def test_product_module():
    """Test the product recommendation module"""
    print("=" * 70)
    print("TESTING PRODUCT RECOMMENDATION MODULE")
    print("=" * 70)
    
    try:
        product_module = ProductRecommendationModule()
        
        # Test with sample customer profile
        customer_profile = product_module.get_customer_profile()
        
        product, confidence, top_3 = product_module.predict(customer_profile)
        
        print(f"\n📊 Test Results:")
        print(f"   Recommended Product: {product}")
        print(f"   Confidence: {confidence:.2%}")
        print(f"\n   Top 3 Recommendations:")
        for i, (prod, conf) in enumerate(top_3, 1):
            print(f"      {i}. {prod} ({conf:.2%})")
        
        print("\n✅ Product module test complete!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")


if __name__ == "__main__":
    test_product_module()