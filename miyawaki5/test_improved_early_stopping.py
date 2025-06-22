#!/usr/bin/env python3
"""
Test Improved Early Stopping Logic
Compare old vs new early stopping behavior
"""

import numpy as np
import matplotlib.pyplot as plt

def old_early_stopping_check(test_losses, epoch):
    """Original early stopping logic"""
    if epoch > 20 and len(test_losses) > 10:
        recent_losses = test_losses[-10:]
        if all(recent_losses[i] >= recent_losses[i+1] for i in range(len(recent_losses)-1)):
            return True, f"Monotonic decrease for 10 epochs"
    return False, "Continuing"

def new_early_stopping_check(test_losses, epoch):
    """Improved early stopping logic"""
    if epoch > 30 and len(test_losses) > 15:
        recent_losses = test_losses[-15:]
        
        if len(recent_losses) >= 10:
            improvement_threshold = 0.001  # 0.1% improvement threshold
            recent_10 = recent_losses[-10:]
            max_recent = max(recent_10)
            min_recent = min(recent_10)
            relative_improvement = (max_recent - min_recent) / max_recent
            
            # Also check if loss has plateaued
            last_5_avg = np.mean(recent_losses[-5:])
            prev_5_avg = np.mean(recent_losses[-10:-5])
            plateau_improvement = (prev_5_avg - last_5_avg) / prev_5_avg
            
            if relative_improvement < improvement_threshold and plateau_improvement < improvement_threshold:
                return True, f"Minimal improvement: {relative_improvement:.4f}, Plateau: {plateau_improvement:.4f}"
    
    return False, "Continuing"

def simulate_training_scenarios():
    """Simulate different training scenarios"""
    
    scenarios = {
        "Scenario 1: Continuous Improvement": {
            # Loss continues to decrease significantly
            "losses": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.32, 0.3, 0.28, 0.26, 0.24, 0.22, 0.2, 0.18, 0.16, 0.14, 0.12, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005],
            "description": "Model still improving significantly"
        },
        
        "Scenario 2: Plateauing": {
            # Loss plateaus after initial decrease
            "losses": [1.0, 0.8, 0.6, 0.4, 0.3, 0.25, 0.24, 0.239, 0.238, 0.237, 0.236, 0.235, 0.234, 0.233, 0.232, 0.231, 0.2309, 0.2308, 0.2307, 0.2306, 0.2305, 0.2304, 0.2303, 0.2302, 0.2301, 0.23009, 0.23008, 0.23007, 0.23006, 0.23005, 0.23004, 0.23003, 0.23002, 0.23001, 0.23000],
            "description": "Model has plateaued, minimal improvement"
        },
        
        "Scenario 3: Miyawaki-like": {
            # Similar to our actual Miyawaki training
            "losses": [1.0928, 0.9729, 0.8781, 0.8050, 0.7434, 0.6939, 0.6547, 0.6167, 0.6089, 0.5632, 0.5432, 0.5176, 0.5232, 0.4992, 0.4874, 0.4967, 0.4948, 0.4503, 0.4502, 0.4368, 0.4423, 0.4543, 0.4200, 0.4100, 0.4050, 0.4000, 0.3950, 0.3900, 0.3850, 0.3800, 0.3750, 0.3700, 0.3650, 0.3600, 0.3550],
            "description": "Real Miyawaki-like training pattern"
        }
    }
    
    results = {}
    
    for scenario_name, scenario_data in scenarios.items():
        losses = scenario_data["losses"]
        description = scenario_data["description"]
        
        print(f"\n🔍 {scenario_name}")
        print(f"   {description}")
        print("=" * 60)
        
        old_stop_epoch = None
        new_stop_epoch = None
        
        # Test both early stopping methods
        for epoch in range(len(losses)):
            current_losses = losses[:epoch+1]
            
            # Test old method
            if old_stop_epoch is None:
                old_stop, old_reason = old_early_stopping_check(current_losses, epoch)
                if old_stop:
                    old_stop_epoch = epoch + 1
                    print(f"   🛑 OLD Early Stop: Epoch {old_stop_epoch} - {old_reason}")
                    print(f"      Final loss: {current_losses[-1]:.4f}")
            
            # Test new method
            if new_stop_epoch is None:
                new_stop, new_reason = new_early_stopping_check(current_losses, epoch)
                if new_stop:
                    new_stop_epoch = epoch + 1
                    print(f"   🛑 NEW Early Stop: Epoch {new_stop_epoch} - {new_reason}")
                    print(f"      Final loss: {current_losses[-1]:.4f}")
            
            # If both stopped, break
            if old_stop_epoch and new_stop_epoch:
                break
        
        # If no early stopping triggered
        if old_stop_epoch is None:
            old_stop_epoch = len(losses)
            print(f"   ✅ OLD: No early stopping, ran full {len(losses)} epochs")
            print(f"      Final loss: {losses[-1]:.4f}")
            
        if new_stop_epoch is None:
            new_stop_epoch = len(losses)
            print(f"   ✅ NEW: No early stopping, ran full {len(losses)} epochs")
            print(f"      Final loss: {losses[-1]:.4f}")
        
        # Calculate potential loss if continued
        old_final_loss = losses[old_stop_epoch-1] if old_stop_epoch <= len(losses) else losses[-1]
        new_final_loss = losses[new_stop_epoch-1] if new_stop_epoch <= len(losses) else losses[-1]
        actual_final_loss = losses[-1]
        
        old_missed_improvement = (old_final_loss - actual_final_loss) / old_final_loss * 100
        new_missed_improvement = (new_final_loss - actual_final_loss) / new_final_loss * 100
        
        print(f"\n   📊 Comparison:")
        print(f"      Old method: Stopped at epoch {old_stop_epoch}, loss {old_final_loss:.4f}")
        print(f"      New method: Stopped at epoch {new_stop_epoch}, loss {new_final_loss:.4f}")
        print(f"      Actual final: Epoch {len(losses)}, loss {actual_final_loss:.4f}")
        print(f"      Old missed improvement: {old_missed_improvement:.2f}%")
        print(f"      New missed improvement: {new_missed_improvement:.2f}%")
        
        results[scenario_name] = {
            'losses': losses,
            'old_stop': old_stop_epoch,
            'new_stop': new_stop_epoch,
            'old_final': old_final_loss,
            'new_final': new_final_loss,
            'actual_final': actual_final_loss,
            'old_missed': old_missed_improvement,
            'new_missed': new_missed_improvement
        }
    
    return results

def visualize_results(results):
    """Visualize early stopping comparison"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (scenario_name, data) in enumerate(results.items()):
        ax = axes[i]
        
        losses = data['losses']
        old_stop = data['old_stop']
        new_stop = data['new_stop']
        
        epochs = range(1, len(losses) + 1)
        
        # Plot loss curve
        ax.plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
        
        # Mark early stopping points
        if old_stop <= len(losses):
            ax.axvline(x=old_stop, color='red', linestyle='--', alpha=0.7, label=f'Old Stop (Epoch {old_stop})')
            ax.scatter([old_stop], [losses[old_stop-1]], color='red', s=100, zorder=5)
        
        if new_stop <= len(losses):
            ax.axvline(x=new_stop, color='green', linestyle='--', alpha=0.7, label=f'New Stop (Epoch {new_stop})')
            ax.scatter([new_stop], [losses[new_stop-1]], color='green', s=100, zorder=5)
        
        ax.set_title(scenario_name.replace("Scenario ", ""), fontsize=12)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('early_stopping_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to test early stopping improvements"""
    
    print("🔍 EARLY STOPPING COMPARISON ANALYSIS")
    print("=" * 70)
    print("Testing old vs new early stopping logic on different scenarios")
    print("=" * 70)
    
    # Run simulations
    results = simulate_training_scenarios()
    
    # Visualize results
    visualize_results(results)
    
    print(f"\n🎯 SUMMARY:")
    print("=" * 50)
    
    for scenario_name, data in results.items():
        print(f"\n{scenario_name}:")
        print(f"   Old method: {data['old_missed']:.2f}% missed improvement")
        print(f"   New method: {data['new_missed']:.2f}% missed improvement")
        
        if data['new_missed'] < data['old_missed']:
            print(f"   ✅ New method is BETTER (less missed improvement)")
        elif data['new_missed'] > data['old_missed']:
            print(f"   ⚠️  Old method was better for this case")
        else:
            print(f"   ➖ Both methods equivalent")
    
    print(f"\n📊 Key Improvements in New Method:")
    print(f"   ✅ Waits longer (30 epochs vs 20)")
    print(f"   ✅ Uses improvement threshold (0.1%)")
    print(f"   ✅ Checks plateau behavior")
    print(f"   ✅ More sophisticated stopping criteria")
    print(f"   ✅ Less likely to stop prematurely")

if __name__ == "__main__":
    main()
