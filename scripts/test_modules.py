"""
Module Testing Script
Tests each module independently before integration
"""

from pathlib import Path
import sys


def test_all_modules():
    """Test all three modules independently"""
    
    print("=" * 70)
    print("MODULE TESTING SUITE")
    print("=" * 70)
    
    results = {
        'face': False,
        'voice': False,
        'product': False
    }
    
    # Test Face Module
    print("\n[1/3] Testing Face Recognition Module...")
    print("-" * 70)
    try:
        from face_module import FaceRecognitionModule
        face_module = FaceRecognitionModule()
        print("✅ Face Recognition Module loaded successfully")
        results['face'] = True
    except Exception as e:
        print(f"❌ Face Recognition Module failed: {str(e)}")
    
    # Test Voice Module
    print("\n[2/3] Testing Voice Verification Module...")
    print("-" * 70)
    try:
        from voice_module import VoiceVerificationModule
        voice_module = VoiceVerificationModule()
        print("✅ Voice Verification Module loaded successfully")
        results['voice'] = True
    except Exception as e:
        print(f"❌ Voice Verification Module failed: {str(e)}")
    
    # Test Product Module
    print("\n[3/3] Testing Product Recommendation Module...")
    print("-" * 70)
    try:
        from product_module import ProductRecommendationModule
        product_module = ProductRecommendationModule()
        print("✅ Product Recommendation Module loaded successfully")
        results['product'] = True
    except Exception as e:
        print(f"❌ Product Recommendation Module failed: {str(e)}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\nPassed: {passed}/{total} modules")
    
    for module, status in results.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {module.upper()} Module: {'PASS' if status else 'FAIL'}")
    
    if passed == total:
        print("\n🎉 ALL MODULES PASSED!")
        print("   System is ready for full integration")
        print("   Run: python scripts/main_system.py")
        return True
    else:
        print("\n⚠️  SOME MODULES FAILED")
        print("   Please fix the issues above before running main system")
        return False


def check_required_files():
    """Check if all required model files exist"""
    
    print("\n" + "=" * 70)
    print("CHECKING REQUIRED FILES")
    print("=" * 70)
    
    required_files = [
        'models/face_recognition_model.pkl',
        'models/voice_verification_model.pkl',
        'models/product_recommendation_model.pkl',
        'models/scaler_face.pkl',
        'models/scaler_voice.pkl',
        'models/scaler_product.pkl',
        'models/label_encoder_face.pkl',
        'models/label_encoder_voice.pkl',
        'models/label_encoder_product.pkl',
        'data/processed/merged_dataset.csv'
    ]
    
    all_exist = True
    
    for file in required_files:
        exists = Path(file).exists()
        icon = "✅" if exists else "❌"
        print(f"{icon} {file}")
        if not exists:
            all_exist = False
    
    print("\n" + "=" * 70)
    
    if all_exist:
        print("✅ All required files present")
    else:
        print("❌ Some files are missing")
        print("\nPlease ensure teammates have provided:")
        print("  • All model .pkl files")
        print("  • All scaler .pkl files")
        print("  • All label_encoder .pkl files")
        print("  • merged_dataset.csv")
    
    return all_exist


def main():
    """Run all checks"""
    print("\n" + "🔍" * 35)
    print("SYSTEM READINESS CHECK")
    print("🔍" * 35 + "\n")
    
    # Check files
    files_ok = check_required_files()
    
    # Test modules
    modules_ok = test_all_modules()
    
    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    
    if files_ok and modules_ok:
        print("\n✅ SYSTEM IS READY!")
        print("   All checks passed")
        print("   You can now run: python scripts/main_system.py")
        return True
    else:
        print("\n❌ SYSTEM NOT READY")
        if not files_ok:
            print("   • Missing required files")
        if not modules_ok:
            print("   • Module initialization failed")
        print("\n   Fix the issues above and run this test again")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)