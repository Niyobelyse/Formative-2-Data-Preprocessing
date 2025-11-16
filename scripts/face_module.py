"""
Face Recognition Module
Handles facial authentication using trained model
"""

import cv2
import numpy as np
import joblib
from pathlib import Path
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
import warnings
warnings.filterwarnings('ignore')


class FaceRecognitionModule:
    """
    Face Recognition System
    Authenticates users based on facial features
    """
    
    def __init__(self, model_dir='models'):
        """
        Initialize face recognition module
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = Path(model_dir)
        
        print("Loading Face Recognition Model...")
        
        try:
            # Load trained model
            self.model = joblib.load(self.model_dir / 'face_recognition_model.pkl')
            self.scaler = joblib.load(self.model_dir / 'scaler_face.pkl')
            self.label_encoder = joblib.load(self.model_dir / 'label_encoder_face.pkl')
            
            # Load VGG16 for feature extraction
            self.feature_extractor = VGG16(weights='imagenet', 
                                          include_top=False, 
                                          pooling='avg')
            
            print("✓ Face Recognition Model loaded successfully\n")
            
        except FileNotFoundError as e:
            print(f"❌ Error: Model files not found in {self.model_dir}/")
            print(f"   Missing: {e.filename}")
            print("   Please ensure teammates have provided model files.")
            raise
    
    def extract_features(self, image_path):
        """
        Extract features from face image
        
        Args:
            image_path: Path to face image
            
        Returns:
            Feature vector
        """
        try:
            # Load and preprocess image
            img = keras_image.load_img(image_path, target_size=(224, 224))
            img_array = keras_image.img_to_array(img)
            img_expanded = np.expand_dims(img_array, axis=0)
            img_preprocessed = preprocess_input(img_expanded)
            
            # Extract VGG16 features
            vgg_features = self.feature_extractor.predict(img_preprocessed, verbose=0).flatten()
            
            # Extract color histograms
            img_cv = cv2.imread(str(image_path))
            img_resized = cv2.resize(img_cv, (224, 224))
            
            hist_r = cv2.calcHist([img_resized], [0], None, [64], [0, 256]).flatten()
            hist_g = cv2.calcHist([img_resized], [1], None, [64], [0, 256]).flatten()
            hist_b = cv2.calcHist([img_resized], [2], None, [64], [0, 256]).flatten()
            
            # Combine features
            features = np.concatenate([vgg_features, hist_r, hist_g, hist_b])
            
            # Ensure correct shape
            expected_features = self.scaler.mean_.shape[0]
            if len(features) < expected_features:
                features = np.pad(features, (0, expected_features - len(features)))
            elif len(features) > expected_features:
                features = features[:expected_features]
            
            return features.reshape(1, -1)
            
        except Exception as e:
            print(f"❌ Error extracting features: {str(e)}")
            return None
    
    def authenticate(self, image_path, confidence_threshold=0.7):
        """
        Authenticate user from face image
        
        Args:
            image_path: Path to face image
            confidence_threshold: Minimum confidence required (default: 0.7)
            
        Returns:
            tuple: (is_authenticated, member_id, confidence)
        """
        print(f"🔍 Analyzing facial features...")
        
        # Extract features
        features = self.extract_features(image_path)
        
        if features is None:
            return False, None, 0.0
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)
        
        # Get member ID
        member_id = self.label_encoder.inverse_transform([prediction])[0]
        
        # Check threshold
        is_authenticated = confidence >= confidence_threshold
        
        return is_authenticated, member_id, confidence
    
    def capture_from_webcam(self, save_path='temp/captured_face.jpg'):
        """
        Capture face from webcam
        
        Args:
            save_path: Where to save captured image
            
        Returns:
            Path to saved image or None if cancelled
        """
        print("\n📸 Face Capture Mode")
        print("   Position your face in the frame")
        print("   Press SPACE to capture, ESC to cancel")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not access webcam")
            return None
        
        # Create temp directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error: Could not read frame")
                break
            
            # Add instructions to frame
            cv2.putText(frame, "Press SPACE to capture, ESC to cancel", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Face Capture', frame)
            
            key = cv2.waitKey(1)
            
            if key == 27:  # ESC
                print("   Capture cancelled")
                cap.release()
                cv2.destroyAllWindows()
                return None
                
            elif key == 32:  # SPACE
                cv2.imwrite(save_path, frame)
                print(f"   ✓ Face captured and saved to {save_path}")
                cap.release()
                cv2.destroyAllWindows()
                return save_path
        
        cap.release()
        cv2.destroyAllWindows()
        return None


def test_face_module():
    """Test the face recognition module"""
    print("=" * 70)
    print("TESTING FACE RECOGNITION MODULE")
    print("=" * 70)
    
    try:
        face_module = FaceRecognitionModule()
        
        # Test with a sample image (you'll need to provide one)
        test_image = 'test_data/authorized/test_face.jpg'
        
        if Path(test_image).exists():
            is_auth, member_id, confidence = face_module.authenticate(test_image)
            
            print(f"\n📊 Test Results:")
            print(f"   Authenticated: {'✅ YES' if is_auth else '❌ NO'}")
            print(f"   Member ID: {member_id}")
            print(f"   Confidence: {confidence:.2%}")
        else:
            print(f"\n⚠️  Test image not found: {test_image}")
            print("   Create test_data/authorized/ folder and add test_face.jpg")
        
        print("\n✅ Face module test complete!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")


if __name__ == "__main__":
    test_face_module()