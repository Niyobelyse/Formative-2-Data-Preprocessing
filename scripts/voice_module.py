"""
Voice Verification Module
Handles voice authentication using trained model
"""

import numpy as np
import librosa
import soundfile as sf
import sounddevice as sd
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class VoiceVerificationModule:
    """
    Voice Verification System
    Confirms user identity through voice authentication
    """
    
    def __init__(self, model_dir='models'):
        """
        Initialize voice verification module
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = Path(model_dir)
        
        print("Loading Voice Verification Model...")
        
        try:
            # Load trained model
            self.model = joblib.load(self.model_dir / 'voice_verification_model.pkl')
            self.scaler = joblib.load(self.model_dir / 'scaler_voice.pkl')
            self.label_encoder = joblib.load(self.model_dir / 'label_encoder_voice.pkl')
            
            print("✓ Voice Verification Model loaded successfully\n")
            
        except FileNotFoundError as e:
            print(f"❌ Error: Model files not found in {self.model_dir}/")
            print(f"   Missing: {e.filename}")
            print("   Please ensure teammates have provided model files.")
            raise
    
    def extract_features(self, audio_path):
        """
        Extract features from audio sample
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Feature vector
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            
            # Delta MFCCs
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
            mfcc_delta_std = np.std(mfcc_delta, axis=1)
            
            # Delta-Delta MFCCs
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            mfcc_delta2_mean = np.mean(mfcc_delta2, axis=1)
            mfcc_delta2_std = np.std(mfcc_delta2, axis=1)
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            
            # Energy features
            rms = librosa.feature.rms(y=y)
            zcr = librosa.feature.zero_crossing_rate(y)
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            chroma_std = np.std(chroma, axis=1)
            
            # Spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            contrast_mean = np.mean(spectral_contrast, axis=1)
            contrast_std = np.std(spectral_contrast, axis=1)
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Combine features
            features = np.concatenate([
                [tempo],
                [np.mean(rms)], [np.std(rms)],
                [np.mean(zcr)], [np.std(zcr)],
                [np.mean(spectral_centroid)], [np.std(spectral_centroid)],
                [np.mean(spectral_rolloff)], [np.std(spectral_rolloff)],
                [np.mean(spectral_bandwidth)], [np.std(spectral_bandwidth)],
                mfcc_mean, mfcc_std,
                mfcc_delta_mean, mfcc_delta_std,
                mfcc_delta2_mean, mfcc_delta2_std,
                chroma_mean, chroma_std,
                contrast_mean, contrast_std
            ])
            
            # Ensure correct shape
            expected_features = self.scaler.mean_.shape[0]
            if len(features) < expected_features:
                features = np.pad(features, (0, expected_features - len(features)))
            elif len(features) > expected_features:
                features = features[:expected_features]
            
            return features.reshape(1, -1)
            
        except Exception as e:
            print(f"❌ Error extracting audio features: {str(e)}")
            return None
    
    def verify(self, audio_path, expected_member_id=None, confidence_threshold=0.7):
        """
        Verify voice sample
        
        Args:
            audio_path: Path to audio file
            expected_member_id: Expected member ID (for cross-validation)
            confidence_threshold: Minimum confidence required
            
        Returns:
            tuple: (is_verified, member_id, confidence)
        """
        print(f"🔍 Analyzing voice pattern...")
        
        # Extract features
        features = self.extract_features(audio_path)
        
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
        
        # Check against expected member if provided
        if expected_member_id is not None:
            is_verified = (str(member_id) == str(expected_member_id)) and (confidence >= confidence_threshold)
        else:
            is_verified = confidence >= confidence_threshold
        
        return is_verified, member_id, confidence
    
    def record_audio(self, duration=3, sample_rate=22050, save_path='temp/recorded_voice.wav'):
        """
        Record audio from microphone
        
        Args:
            duration: Recording duration in seconds
            sample_rate: Audio sample rate
            save_path: Where to save recorded audio
            
        Returns:
            Path to saved audio or None if failed
        """
        print(f"\n🎤 Voice Recording Mode")
        print(f"   Say: 'Yes, approve' or 'Confirm transaction'")
        input("   Press ENTER when ready to record...")
        
        # Create temp directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            print(f"   🔴 RECORDING... ({duration} seconds)")
            
            # Record audio
            audio_data = sd.rec(int(duration * sample_rate),
                               samplerate=sample_rate,
                               channels=1,
                               dtype='float32')
            sd.wait()
            
            print("   ✓ Recording complete!")
            
            # Save audio
            sf.write(save_path, audio_data, sample_rate)
            print(f"   ✓ Audio saved to {save_path}")
            
            return save_path
            
        except Exception as e:
            print(f"❌ Error recording audio: {str(e)}")
            return None


def test_voice_module():
    """Test the voice verification module"""
    print("=" * 70)
    print("TESTING VOICE VERIFICATION MODULE")
    print("=" * 70)
    
    try:
        voice_module = VoiceVerificationModule()
        
        # Test with a sample audio (you'll need to provide one)
        test_audio = 'test_data/authorized/test_voice.wav'
        
        if Path(test_audio).exists():
            is_verified, member_id, confidence = voice_module.verify(test_audio)
            
            print(f"\n📊 Test Results:")
            print(f"   Verified: {'✅ YES' if is_verified else '❌ NO'}")
            print(f"   Member ID: {member_id}")
            print(f"   Confidence: {confidence:.2%}")
        else:
            print(f"\n⚠️  Test audio not found: {test_audio}")
            print("   Create test_data/authorized/ folder and add test_voice.wav")
        
        print("\n✅ Voice module test complete!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")


if __name__ == "__main__":
    test_voice_module()