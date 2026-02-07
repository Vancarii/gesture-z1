import os
import numpy as np

STATIC_GESTURES = ["point_left", "point_right", "point_up", "point_down", 
                   "point_back", "point_forward"]
DYNAMIC_GESTURES = ["move_hand_up", "move_hand_down", "pull_back", 
                    "swipe_back", "swipe_towards", "clap", "pause"]

def validate_collected_data(data_dir="data"):
    print("\n" + "="*70)
    print("VALIDATING COLLECTED DATA")
    print("="*70)
    
    all_issues = []
    
    for gesture in os.listdir(data_dir):
        gesture_path = os.path.join(data_dir, gesture)
        if not os.path.isdir(gesture_path):
            continue
        
        files = sorted([f for f in os.listdir(gesture_path) if f.endswith('.npy')])
        print(f"\n{'='*70}")
        print(f"📁 {gesture}: {len(files)} samples")
        print(f"{'='*70}")
        
        if len(files) == 0:
            print("  ❌ NO SAMPLES FOUND")
            continue
        
        samples = []
        for file in files:
            data = np.load(os.path.join(gesture_path, file))
            samples.append(data)
        
        samples = np.array(samples)  # Shape: (n_samples, 90, n_features)
        n_samples, n_frames, n_features = samples.shape
        
        # 1. Check for NaN/Inf
        has_issues = False
        for i, file in enumerate(files):
            data = samples[i]
            if np.isnan(data).any():
                print(f"  ❌ {file}: Contains NaN values")
                has_issues = True
            elif np.isinf(data).any():
                print(f"  ❌ {file}: Contains Inf values")
                has_issues = True
            elif data.shape[0] != 90:
                print(f"  ❌ {file}: Wrong shape {data.shape}")
                has_issues = True
        
        if has_issues:
            all_issues.append(gesture)
            continue
        
        # 2. Compute statistics across all samples
        # Mean trajectory across time and samples
        mean_trajectory = samples.mean(axis=0)  # (90, n_features)
        
        # Temporal variance (how much each feature changes over time)
        temporal_var = np.var(mean_trajectory, axis=0)  # (n_features,)
        
        # Inter-sample variance (how different are samples from each other)
        sample_means = samples.mean(axis=1)  # (n_samples, n_features)
        inter_sample_var = np.var(sample_means, axis=0)  # (n_features,)
        
        print(f"\n  Feature Analysis:")
        print(f"    - Temporal variance (avg): {temporal_var.mean():.3f}")
        print(f"    - Inter-sample variance (avg): {inter_sample_var.mean():.3f}")
        
        # 3. Detect outlier samples using robust statistics
        # Use median absolute deviation (MAD) - more robust than std
        sample_medians = np.median(sample_means, axis=0)
        mad = np.median(np.abs(sample_means - sample_medians), axis=0) + 1e-9
        
        outlier_threshold = 3.5  # MAD-based threshold
        outliers = []
        
        for i, file in enumerate(files):
            # Compute MAD score for this sample
            mad_scores = np.abs(sample_means[i] - sample_medians) / (mad + 1e-8)
            max_mad = mad_scores.max()
            
            if max_mad > outlier_threshold:
                outliers.append((i, file, max_mad))
        
        if outliers:
            print(f"\n  ⚠️  OUTLIER SAMPLES DETECTED:")
            for idx, file, score in outliers:
                print(f"    - {file}: MAD score = {score:.2f}")
                all_issues.append(f"{gesture}/{file}")
        
        # 4. Check for tracking glitches (sudden jumps in trajectory)
        for i, file in enumerate(files):
            trajectory = samples[i]  # (90, n_features)
            
            # Compute frame-to-frame differences
            diffs = np.diff(trajectory, axis=0)  # (89, n_features)
            diff_norms = np.linalg.norm(diffs, axis=1) 
            
            # Find sudden jumps (>5x median jump size)
            median_jump = np.median(diff_norms) + 1e-9
            max_jump = diff_norms.max()
            
            if max_jump > 5 * median_jump and max_jump > 10:
                jump_frame = diff_norms.argmax()
                print(f"  ⚠️  {file}: Large jump at frame {jump_frame} "
                      f"(magnitude: {max_jump:.1f} vs median: {median_jump:.1f})")
                all_issues.append(f"{gesture}/{file}")
        
        # 5. Gesture-specific checks
        is_static = any(static in gesture for static in STATIC_GESTURES)
        
        if is_static:
            # Static gestures should have low temporal variance
            if temporal_var.mean() > 1.0:
                print(f"  ⚠️  Static gesture has high temporal variance "
                      f"({temporal_var.mean():.1f}) - check if gesture was held still")
        else:
            # Dynamic gestures should have reasonable temporal variance
            if temporal_var.mean() < 0.5 and gesture != "clap":
                print(f"  ⚠️  Dynamic gesture has very low temporal variance "
                      f"({temporal_var.mean():.1f}) - gestures may be too slow/small")
        
        # 6. Check for consistent gesture execution (clustering)
        if inter_sample_var.mean() > 100:
            print(f"  ⚠️  High inter-sample variance ({inter_sample_var.mean():.1f}) "
                  f"- gesture execution may be inconsistent")
            print(f"      Consider reviewing samples for consistency")
        
        if not has_issues and not outliers:
            print(f"\n  ✅ All samples look good!")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*70}")
    if all_issues:
        print(f"❌ Found {len(all_issues)} issues across gestures:")
        for issue in set(all_issues):
            print(f"  - {issue}")
        print(f"\nRecommendation: Re-record flagged samples")
    else:
        print(f"✅ All data looks good! Ready for scaling and HMM training.")
    
    return len(all_issues) == 0

if __name__ == "__main__":
    validate_collected_data()
