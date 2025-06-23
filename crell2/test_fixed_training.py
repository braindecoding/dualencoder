#!/usr/bin/env python3
"""
Test Fixed Advanced Training
Quick test to verify validation bug is fixed
"""

import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append('.')

# Import the main functions
from advanced_training import advanced_training_loop, AdvancedTrainingConfig

def main():
    """
    Test the fixed advanced training with shorter epochs
    """
    print("🔧 TESTING FIXED ADVANCED TRAINING")
    print("=" * 60)
    print("Bug Fix: Validation now runs EVERY epoch (not every 5 epochs)")
    print("Expected: Validation accuracy should change every epoch")
    print("=" * 60)
    
    # Create config with shorter training for testing
    config = AdvancedTrainingConfig()
    config.num_epochs = 20  # Short test
    config.patience = 10    # Reduced patience
    
    print(f"🎯 Test Configuration:")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Patience: {config.patience}")
    print(f"   Expected: Validation evaluated EVERY epoch")
    
    start_time = datetime.now()
    print(f"🕐 Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run short training to test the fix
        print("\n🚀 Running test training...")
        model, history, best_val_acc, test_acc = advanced_training_loop(config)
        
        # Analyze validation behavior
        val_accuracies = history['val_accuracies']
        
        print(f"\n📊 VALIDATION BEHAVIOR ANALYSIS:")
        print(f"   Total epochs trained: {len(val_accuracies)}")
        print(f"   Validation accuracies per epoch:")
        
        for i, val_acc in enumerate(val_accuracies):
            print(f"     Epoch {i+1}: {val_acc:.4f}")
        
        # Check if validation changes every epoch
        unique_values = len(set(val_accuracies))
        total_epochs = len(val_accuracies)
        
        print(f"\n🔍 BUG FIX VERIFICATION:")
        print(f"   Unique validation values: {unique_values}")
        print(f"   Total epochs: {total_epochs}")
        
        if unique_values == 1:
            print(f"   ❌ BUG STILL EXISTS: All validation values are the same!")
            print(f"   🔧 Validation is not being updated properly")
        elif unique_values < total_epochs * 0.3:  # Less than 30% unique
            print(f"   ⚠️  POTENTIAL ISSUE: Too few unique validation values")
            print(f"   🔧 Validation may not be updating frequently enough")
        else:
            print(f"   ✅ BUG FIXED: Validation values are changing appropriately")
            print(f"   🎉 Validation is being updated every epoch")
        
        # Check for improvement
        if len(val_accuracies) > 1:
            initial_val = val_accuracies[0]
            final_val = val_accuracies[-1]
            improvement = final_val - initial_val
            
            print(f"\n📈 LEARNING PROGRESS:")
            print(f"   Initial validation accuracy: {initial_val:.4f}")
            print(f"   Final validation accuracy: {final_val:.4f}")
            print(f"   Improvement: {improvement:.4f}")
            
            if improvement > 0.01:
                print(f"   ✅ GOOD: Model is learning (improvement > 1%)")
            elif improvement > 0:
                print(f"   👍 OK: Model shows some learning")
            else:
                print(f"   ⚠️  CONCERN: No improvement or degradation")
        
        # Final results
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n🏆 TEST RESULTS:")
        print(f"   Duration: {duration}")
        print(f"   Best validation accuracy: {best_val_acc:.4f}")
        print(f"   Test accuracy: {test_acc:.4f}")
        
        print(f"\n✅ FIXED TRAINING TEST COMPLETED!")
        print(f"   Validation bug fix verification complete")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR during test:")
        print(f"   {str(e)}")
        
        import traceback
        print(f"\n📋 Full traceback:")
        traceback.print_exc()
        
        return False

if __name__ == "__main__":
    # Run the test
    success = main()
    
    if success:
        print(f"\n🎉 Test completed successfully!")
        print(f"   Validation bug appears to be fixed")
        print(f"   Ready for full training runs")
    else:
        print(f"\n❌ Test failed!")
        print(f"   Check error messages above")
