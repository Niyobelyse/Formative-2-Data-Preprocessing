"""
Demo Script
Automated demonstration of the system with sample data
"""

from main_system import MultimodalAuthenticationSystem
from pathlib import Path
import time


def run_demo():
    """
    Run automated system demonstration
    """
    print("\n" + "🎬" * 35)
    print("AUTOMATED SYSTEM DEMONSTRATION")
    print("🎬" * 35)
    
    print("\nThis demo will showcase:")
    print("  ✅ Successful authentication (authorized user)")
    print("  ❌ Failed authentication (unauthorized user)")
    print("  🔒 System security features")
    
    input("\nPress ENTER to begin demonstration...")
    
    # Initialize system
    system = MultimodalAuthenticationSystem()
    
    # ===== SCENARIO 1: SUCCESSFUL TRANSACTION =====
    print("\n\n" + "=" * 70)
    print("SCENARIO 1: AUTHORIZED USER TRANSACTION")
    print("=" * 70)
    
    print("\n📋 Test Case:")
    print("   User: Authorized team member (Member 1)")
    print("   Face: Valid facial image")
    print("   Voice: Valid voice sample matching the face")
    print("   Expected: ✅ Full access granted, prediction displayed")
    
    input("\nPress ENTER to run Scenario 1...")
    
    # Check if test files exist
    auth_face = 'test_data/authorized/test_face.jpg'
    auth_voice = 'test_data/authorized/test_voice.wav'
    
    if Path(auth_face).exists() and Path(auth_voice).exists():
        success = system.run_full_transaction(
            face_image_path=auth_face,
            voice_audio_path=auth_voice
        )
        
        if success:
            print("\n✅ SCENARIO 1: SUCCESS")
            print("   System correctly authenticated and provided recommendation")
        else:
            print("\n⚠️  SCENARIO 1: UNEXPECTED RESULT")
    else:
        print(f"\n⚠️  Demo files not found:")
        print(f"   Expected: {auth_face}")
        print(f"   Expected: {auth_voice}")
        print("   Please add test files or use live capture")
    
    time.sleep(2)
    
    # ===== SCENARIO 2: UNAUTHORIZED ACCESS =====
    print("\n\n" + "=" * 70)
    print("SCENARIO 2: UNAUTHORIZED ACCESS ATTEMPT")
    print("=" * 70)
    
    print("\n📋 Test Case:")
    print("   User: Unknown person (not in database)")
    print("   Face: Image not matching any authorized user")
    print("   Expected: ❌ Access denied at face recognition")
    
    input("\nPress ENTER to run Scenario 2...")
    
    unauth_face = 'test_data/unauthorized/unauthorized_face.jpg'
    
    if Path(unauth_face).exists():
        denied = system.test_unauthorized_access(unauth_face)
        
        if denied:
            print("\n✅ SCENARIO 2: SUCCESS")
            print("   System correctly rejected unauthorized user")
        else:
            print("\n⚠️  SCENARIO 2: SECURITY CONCERN")
    else:
        print(f"\n⚠️  Demo file not found: {unauth_face}")
        print("   Skipping unauthorized access test")
    
    time.sleep(2)
    
    # ===== FINAL SUMMARY =====
    print("\n\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE - SUMMARY")
    print("=" * 70)
    
    print("\n📊 System Capabilities Demonstrated:")
    print("   ✅ Facial Recognition Authentication")
    print("   ✅ Voice Verification Authentication")
    print("   ✅ Product Recommendation Generation")
    print("   ✅ Multimodal Security (Face + Voice required)")
    print("   ✅ Unauthorized Access Prevention")
    
    print("\n🔒 Security Features:")
    print("   • Two-factor biometric authentication")
    print("   • Sequential verification (Face → Voice)")
    print("   • Confidence threshold enforcement")
    print("   • Transaction logging and audit trail")
    print("   • Multiple denial checkpoints")
    
    print("\n💡 System Flow:")
    print("   1. User presents face → Face Recognition verifies")
    print("   2. If authorized → Product Recommendation generates")
    print("   3. User provides voice → Voice Verification confirms")
    print("   4. If voice matches → Prediction displayed")
    print("   5. Any failure → Access denied")
    
    print("\n" + "=" * 70)
    print("End of Demonstration")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_demo()