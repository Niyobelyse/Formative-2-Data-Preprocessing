"""
Main System Integration - Command Line Interface
Multimodal Authentication & Product Recommendation System
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import time

# Import custom modules
from face_module import FaceRecognitionModule
from voice_module import VoiceVerificationModule
from product_module import ProductRecommendationModule


class MultimodalAuthenticationSystem:
    """
    Complete Multimodal Authentication System
    Integrates Face Recognition, Voice Verification, and Product Recommendation
    """
    
    def __init__(self):
        """Initialize the complete system"""
        print("\n" + "=" * 70)
        print("MULTIMODAL AUTHENTICATION & RECOMMENDATION SYSTEM")
        print("=" * 70)
        print("\nInitializing system components...")
        print("-" * 70)
        
        try:
            # Initialize modules
            self.face_module = FaceRecognitionModule()
            self.voice_module = VoiceVerificationModule()
            self.product_module = ProductRecommendationModule()
            
            # Create necessary directories
            os.makedirs('temp', exist_ok=True)
            os.makedirs('logs', exist_ok=True)
            
            # Setup logging
            self.log_file = f'logs/system_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
            
            print("=" * 70)
            print("✅ SYSTEM INITIALIZED SUCCESSFULLY!")
            print("=" * 70 + "\n")
            
        except Exception as e:
            print(f"\n❌ SYSTEM INITIALIZATION FAILED!")
            print(f"Error: {str(e)}")
            print("\nPlease ensure:")
            print("  1. All model files are in models/ directory")
            print("  2. merged_dataset.csv is in data/processed/")
            print("  3. All dependencies are installed")
            sys.exit(1)
    
    def log_transaction(self, message):
        """
        Log transaction details
        
        Args:
            message: Message to log
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
        
        # Also print to console (optional, comment out if too verbose)
        # print(log_entry.strip())
    
    def print_header(self, text):
        """Print a formatted header"""
        print("\n" + "=" * 70)
        print(text.center(70))
        print("=" * 70 + "\n")
    
    def print_step(self, step_num, title):
        """Print a step header"""
        print("\n" + "-" * 70)
        print(f"STEP {step_num}: {title}")
        print("-" * 70)
    
    def run_full_transaction(self, use_webcam=False, use_microphone=False,
                           face_image_path=None, voice_audio_path=None):
        """
        Execute complete authentication and recommendation flow
        
        Args:
            use_webcam: Capture face from webcam
            use_microphone: Record voice from microphone
            face_image_path: Path to pre-recorded face image
            voice_audio_path: Path to pre-recorded voice audio
            
        Returns:
            bool: Success status
        """
        self.print_header("STARTING FULL TRANSACTION FLOW")
        self.log_transaction("Transaction initiated")
        
        # ===== STEP 1: FACIAL RECOGNITION =====
        self.print_step(1, "FACIAL RECOGNITION")
        
        # Get face input
        if use_webcam:
            print("📸 Capturing face from webcam...")
            face_path = self.face_module.capture_from_webcam()
            if face_path is None:
                print("\n❌ Face capture cancelled")
                self.log_transaction("FAILED: Face capture cancelled")
                return False
        elif face_image_path and Path(face_image_path).exists():
            face_path = face_image_path
            print(f"📁 Using provided image: {face_path}")
        else:
            print("❌ No face input provided")
            self.log_transaction("FAILED: No face input")
            return False
        
        # Authenticate face
        face_auth, member_id, face_confidence = self.face_module.authenticate(face_path)
        
        # Display results
        print(f"\n📊 Facial Recognition Results:")
        print(f"   Status: {'✅ AUTHORIZED' if face_auth else '❌ DENIED'}")
        print(f"   Detected Member: Member {member_id}")
        print(f"   Confidence: {face_confidence:.2%}")
        
        self.log_transaction(f"Face Auth - Member: {member_id}, Confidence: {face_confidence:.2%}, Status: {face_auth}")
        
        if not face_auth:
            self.print_header("❌ ACCESS DENIED - FACE NOT RECOGNIZED")
            self.log_transaction("Transaction DENIED at facial recognition")
            return False
        
        print(f"\n✅ Face verified! Welcome, Member {member_id}")
        
        # ===== STEP 2: PRODUCT RECOMMENDATION =====
        self.print_step(2, "PRODUCT RECOMMENDATION")
        
        print(f"🔍 Generating personalized recommendation for Member {member_id}...")
        
        # Get customer profile
        customer_profile = self.product_module.get_customer_profile(member_id=member_id)
        
        # Get prediction
        product, product_confidence, top_3 = self.product_module.predict(customer_profile)
        
        # Display results
        print(f"\n📊 Product Recommendation Results:")
        print(f"   Recommended Product: {product}")
        print(f"   Confidence: {product_confidence:.2%}")
        print(f"\n   Top 3 Recommendations:")
        for i, (prod, conf) in enumerate(top_3, 1):
            print(f"      {i}. {prod} ({conf:.2%})")
        
        self.log_transaction(f"Product Recommended: {product} (Confidence: {product_confidence:.2%})")
        
        # ===== STEP 3: VOICE VERIFICATION =====
        self.print_step(3, "VOICE VERIFICATION")
        
        print(f"🎤 Voice verification required to approve recommendation...")
        
        # Get voice input
        if use_microphone:
            print("\n📢 Please say: 'Yes, approve' or 'Confirm transaction'")
            voice_path = self.voice_module.record_audio()
            if voice_path is None:
                print("\n❌ Voice recording failed")
                self.log_transaction("FAILED: Voice recording failed")
                return False
        elif voice_audio_path and Path(voice_audio_path).exists():
            voice_path = voice_audio_path
            print(f"📁 Using provided audio: {voice_path}")
        else:
            print("❌ No voice input provided")
            self.log_transaction("FAILED: No voice input")
            return False
        
        # Verify voice
        voice_verified, voice_member_id, voice_confidence = self.voice_module.verify(
            voice_path, 
            expected_member_id=member_id
        )
        
        # Display results
        print(f"\n📊 Voice Verification Results:")
        print(f"   Status: {'✅ VERIFIED' if voice_verified else '❌ DENIED'}")
        print(f"   Detected Member: Member {voice_member_id}")
        print(f"   Expected Member: Member {member_id}")
        print(f"   Confidence: {voice_confidence:.2%}")
        print(f"   Match: {'✅ YES' if str(voice_member_id) == str(member_id) else '❌ NO'}")
        
        self.log_transaction(f"Voice Verify - Member: {voice_member_id}, Confidence: {voice_confidence:.2%}, Status: {voice_verified}")
        
        if not voice_verified:
            self.print_header("❌ ACCESS DENIED - VOICE VERIFICATION FAILED")
            self.log_transaction("Transaction DENIED at voice verification")
            return False
        
        # ===== FINAL STEP: DISPLAY APPROVED PREDICTION =====
        self.print_header("✅ TRANSACTION APPROVED - ALL AUTHENTICATION PASSED")
        
        print(f"🎉 Prediction Approved for Member {member_id}")
        print(f"\n📦 YOUR RECOMMENDED PRODUCT:")
        print(f"   ╔{'═' * 50}╗")
        print(f"   ║  {product.upper().center(48)}  ║")
        print(f"   ╚{'═' * 50}╝")
        print(f"\n   Confidence: {product_confidence:.2%}")
        print(f"\n💡 Alternative Options:")
        for i, (prod, conf) in enumerate(top_3[1:], 1):
            print(f"   {i}. {prod} ({conf:.2%})")
        
        self.log_transaction(f"Transaction APPROVED - Product: {product}")
        
        print("\n" + "=" * 70)
        print("Thank you for using our secure recommendation system!")
        print("=" * 70 + "\n")
        
        return True
    
    def test_unauthorized_access(self, unauthorized_face_path):
        """
        Test system with unauthorized user
        
        Args:
            unauthorized_face_path: Path to unauthorized face image
            
        Returns:
            bool: True if correctly denied, False if incorrectly authorized
        """
        self.print_header("TESTING UNAUTHORIZED ACCESS SCENARIO")
        self.log_transaction("Unauthorized access test initiated")
        
        print("🔍 Testing with unauthorized face image...")
        
        if not Path(unauthorized_face_path).exists():
            print(f"❌ File not found: {unauthorized_face_path}")
            return False
        
        # Attempt authentication
        face_auth, member_id, face_confidence = self.face_module.authenticate(unauthorized_face_path)
        
        # Display results
        print(f"\n📊 Security Test Results:")
        print(f"   Face Authenticated: {'❌ YES (SECURITY BREACH!)' if face_auth else '✅ NO (CORRECT)'}")
        print(f"   Detected as: {f'Member {member_id}' if member_id else 'Unknown'}")
        print(f"   Confidence: {face_confidence:.2%}")
        
        if not face_auth:
            print("\n✅ SECURITY TEST PASSED")
            print("   System correctly denied unauthorized access")
            self.log_transaction("Security test PASSED - Unauthorized access denied")
            success = True
        else:
            print("\n⚠️  SECURITY TEST FAILED")
            print("   System incorrectly authorized unknown user!")
            self.log_transaction("Security test FAILED - False positive")
            success = False
        
        print("=" * 70 + "\n")
        
        return success
    
    def display_menu(self):
        """Display the main menu"""
        print("\n" + "=" * 70)
        print("MULTIMODAL AUTHENTICATION SYSTEM - MAIN MENU")
        print("=" * 70)
        print("\n1. Run Full Transaction (Live - Webcam + Microphone)")
        print("2. Run Full Transaction (Pre-recorded Files)")
        print("3. Test Unauthorized Access")
        print("4. Test Individual Modules")
        print("5. View System Logs")
        print("6. Exit")
        print("\n" + "=" * 70)
    
    def test_individual_modules(self):
        """Test each module individually"""
        self.print_header("INDIVIDUAL MODULE TESTING")
        
        while True:
            print("\nSelect module to test:")
            print("1. Face Recognition Module")
            print("2. Voice Verification Module")
            print("3. Product Recommendation Module")
            print("4. Back to Main Menu")
            
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == '1':
                print("\n--- Face Recognition Test ---")
                face_path = input("Enter path to face image: ").strip()
                if Path(face_path).exists():
                    is_auth, member_id, conf = self.face_module.authenticate(face_path)
                    print(f"\nResult: {'✅ Authorized' if is_auth else '❌ Denied'}")
                    print(f"Member: {member_id}, Confidence: {conf:.2%}")
                else:
                    print("❌ File not found")
            
            elif choice == '2':
                print("\n--- Voice Verification Test ---")
                voice_path = input("Enter path to audio file: ").strip()
                if Path(voice_path).exists():
                    is_verified, member_id, conf = self.voice_module.verify(voice_path)
                    print(f"\nResult: {'✅ Verified' if is_verified else '❌ Denied'}")
                    print(f"Member: {member_id}, Confidence: {conf:.2%}")
                else:
                    print("❌ File not found")
            
            elif choice == '3':
                print("\n--- Product Recommendation Test ---")
                profile = self.product_module.get_customer_profile()
                product, conf, top_3 = self.product_module.predict(profile)
                print(f"\nRecommended: {product} ({conf:.2%})")
                print(f"Top 3: {[p[0] for p in top_3]}")
            
            elif choice == '4':
                break
            
            else:
                print("❌ Invalid choice")
    
    def view_logs(self):
        """Display recent system logs"""
        self.print_header("SYSTEM LOGS")
        
        if not Path(self.log_file).exists():
            print("No logs available yet")
            return
        
        with open(self.log_file, 'r') as f:
            logs = f.readlines()
        
        print(f"Showing last 20 log entries from {self.log_file}:\n")
        for log in logs[-20:]:
            print(log.strip())
        
        print("\n" + "=" * 70)
    
    def run_interactive_menu(self):
        """Run the interactive command-line menu"""
        while True:
            self.display_menu()
            
            choice = input("Enter your choice (1-6): ").strip()
            
            if choice == '1':
                # Live capture
                print("\n🚀 Starting live transaction with webcam and microphone...")
                input("Press ENTER to continue...")
                self.run_full_transaction(use_webcam=True, use_microphone=True)
            
            elif choice == '2':
                # Pre-recorded files
                print("\n📁 Using pre-recorded files...")
                face_path = input("Enter path to face image: ").strip()
                voice_path = input("Enter path to voice audio: ").strip()
                
                if Path(face_path).exists() and Path(voice_path).exists():
                    self.run_full_transaction(
                        face_image_path=face_path,
                        voice_audio_path=voice_path
                    )
                else:
                    print("❌ One or more files not found!")
            
            elif choice == '3':
                # Unauthorized access test
                print("\n🔒 Testing security with unauthorized access...")
                unauth_face = input("Enter path to unauthorized face image: ").strip()
                
                if Path(unauth_face).exists():
                    self.test_unauthorized_access(unauth_face)
                else:
                    print("❌ File not found!")
            
            elif choice == '4':
                # Test individual modules
                self.test_individual_modules()
            
            elif choice == '5':
                # View logs
                self.view_logs()
            
            elif choice == '6':
                # Exit
                print("\n👋 Thank you for using the system. Goodbye!")
                break
            
            else:
                print("\n❌ Invalid choice. Please enter 1-6.")
            
            input("\nPress ENTER to continue...")


def main():
    """Main entry point"""
    try:
        # Initialize system
        system = MultimodalAuthenticationSystem()
        
        # Run interactive menu
        system.run_interactive_menu()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  System interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()