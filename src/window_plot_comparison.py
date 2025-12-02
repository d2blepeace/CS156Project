import random
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from windowing import SlidingWindow
from ml_models.random_forest import load_demo_model
from data_loader import BASE_DIR, load_kuhar_subsamples
from constants import CLASS_NAMES

def extract_features_from_window(window_data):
    data_array = np.array(window_data)
    features = []
    for axis_idx in range(3):
        axis_data = data_array[:, axis_idx]
        _mean = np.mean(axis_data)
        _std = np.std(axis_data)
        _min = np.min(axis_data)
        _max = np.max(axis_data)
        _median = np.median(axis_data)
        _energy = np.mean(axis_data**2)
        features.extend([_mean, _std, _min, _max, _median, _energy])
    return np.array(features).reshape(1, -1)

def test_window_size_accuracy(window_size, model, test_data, test_labels, 
                               window_step_size=None, verbose=False):
    if window_step_size is None:
        window_step_size = window_size

    # Limit data to first N samples for speed
    max_test_samples = 2000
    test_subset = test_data[:max_test_samples]
    labels_subset = test_labels[:len(test_subset)]
    
    window = SlidingWindow(
        window_size=window_size,
        step_size=window_step_size,
        padding_strategy='none',
    )
    
    predictions = []
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    # Add all test data to window
    for sample_idx, sample in enumerate(test_subset):
        window.add_sample(sample)
        
        if window.is_ready:
            try:
                window_array = window.get_current_window()
                features = extract_features_from_window(window_array)
                pred = model.predict(features)[0]
                predictions.append(pred)
                all_preds.append(pred)
                
                # FIX: Use the label of the CURRENT sample (end of window)
                # This is the most recent sample that was just added
                if sample_idx < len(labels_subset):
                    true_label = labels_subset[sample_idx]
                    all_labels.append(true_label)
                    
                    if pred == true_label:
                        correct += 1
                    total += 1
                
                window.advance()
            except Exception as e:
                if verbose:
                    print(f"Error at sample {sample_idx}: {e}")
    
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # Calculate F1 score (weighted average for multi-class)
    f1 = 0.0
    if len(all_preds) > 0 and len(all_labels) > 0:
        try:
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        except:
            f1 = 0.0
    
    return {
        'window_size': window_size,
        'accuracy': accuracy,
        'f1_score': f1,
        'predictions': len(predictions),
        'correct': correct,
        'total': total,
    }

# seed_number = random.randint(0, 10000)
def load_demo_test_data(n_samples=500):
    print("Loading demo dataset (Stand/Walk/Jump)...")
    np.random.seed(6376)
    try:
        subsamples = load_kuhar_subsamples(BASE_DIR)
        
        # Filter to demo classes
        demo_classes = ["Stand", "Walk", "Jump"]
        demo_data = subsamples[subsamples['class_name'].isin(demo_classes)].copy()
        
        if len(demo_data) == 0:
            raise RuntimeError("No demo data found in KU-HAR subsamples for Stand/Walk/Jump.")
        
        # Convert to list of samples
        sensor_cols = [c for c in demo_data.columns if c not in 
                      ['subject', 'window_len', 'class_idx', 'class_name', 'serial_no']]
        
        data = []
        labels = []
        
        for idx, row in demo_data.iterrows():
            sensor_values = row[sensor_cols].values.astype(float)
            class_name = row['class_name']
            label = demo_classes.index(class_name)
            
            # Split into 3 axes (accX, accY, accZ interleaved)
            for i in range(0, len(sensor_values) - 2, 3):
                x = sensor_values[i]
                y = sensor_values[i+1] if i+1 < len(sensor_values) else 0
                z = sensor_values[i+2] if i+2 < len(sensor_values) else 0
                data.append([x, y, z])
                labels.append(label)  # Each sample gets the same label
        
        # Limit to n_samples
        if len(data) > n_samples:
            indices = np.random.choice(len(data), n_samples, replace=False)
            data = [data[i] for i in indices]
            labels = [labels[i] for i in indices]
        
        print(f"Loaded {len(data)} samples from dataset")
        print(f"  Class distribution: {np.bincount(labels)}")
        return data, np.array(labels)
        
    except Exception as e:
        print(f"Error loading demo KU-HAR data: {e}")
        raise

def run_accuracy_study():
    """Run accuracy comparison across different window sizes."""
    print("\n" + "="*70)
    print("WINDOW SIZE ACCURACY STUDY (Non-overlapping vs 50% Overlap)")
    print("="*70)
    
    # Load model
    print("\nLoading demo model...")
    try:
        model = load_demo_model()
        print("Demo model loaded (Stand/Walk/Jump)")
    except Exception as e:
        print(f"✗ Could not load model: {e}")
        return None
    
    # Load test data (use fewer samples for faster testing)
    data, labels = load_demo_test_data(n_samples=500)
    print(f"Loaded {len(data)} test samples with {len(np.unique(labels))} classes")
    
    # Test different window sizes
    window_sizes = [10, 20, 30, 40, 50, 75, 100, 150, 200]
    results_no_overlap = []
    results_50_overlap = []
    
    print("\nTesting window sizes (Non-overlapping):")
    print("-" * 70)
    for ws in window_sizes:
        metrics = test_window_size_accuracy(ws, model, data, labels, window_step_size=ws)
        results_no_overlap.append(metrics)
        acc_str = f"{metrics['accuracy']:.1f}%" if metrics['total'] > 0 else "N/A"
        f1_str = f"{metrics['f1_score']:.3f}"
        print(f"Window size {ws:3d}: Accuracy = {acc_str:>6}, F1 = {f1_str:>6} "
              f"({metrics['correct']}/{metrics['total']} predictions)")
    
    print("\nTesting window sizes (50% Overlap):")
    print("-" * 70)
    for ws in window_sizes:
        step_size = max(1, ws // 2)  # 50% overlap
        metrics = test_window_size_accuracy(ws, model, data, labels, window_step_size=step_size)
        results_50_overlap.append(metrics)
        acc_str = f"{metrics['accuracy']:.1f}%" if metrics['total'] > 0 else "N/A"
        f1_str = f"{metrics['f1_score']:.3f}"
        print(f"Window size {ws:3d}: Accuracy = {acc_str:>6}, F1 = {f1_str:>6} "
              f"({metrics['correct']}/{metrics['total']} predictions)")
    
    return results_no_overlap, results_50_overlap

def plot_accuracy_comparison(results_no_overlap, results_50_overlap):
    """Create bar graphs comparing accuracy and F1 score across window sizes with/without overlap."""
    if not results_no_overlap or len(results_no_overlap) == 0:
        print("No results to plot")
        return
    
    # 2x2 layout:
    #   (0,0) Accuracy vs window size (bar)
    #   (0,1) F1 vs window size (bar)
    #   (1,0) Accuracy vs F1 (non-overlapping)
    #   (1,1) Accuracy vs F1 (50% overlap)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'Window Size Performance: Accuracy vs F1 Score\nNon-overlapping vs 50% Overlap',
        fontsize=15,
        fontweight='bold',
    )
    
    window_sizes = [r['window_size'] for r in results_no_overlap]
    
    # --- Plot 1: Accuracy Comparison (Bar Chart) ---
    ax = axes[0, 0]
    x = np.arange(len(window_sizes))
    width = 0.35
    
    acc_no_overlap = [r['accuracy'] for r in results_no_overlap]
    acc_50_overlap = [r['accuracy'] for r in results_50_overlap]
    
    bars1 = ax.bar(x - width/2, acc_no_overlap, width, label='Non-overlapping', 
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, acc_50_overlap, width, label='50% Overlap', 
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Window Size (samples)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(window_sizes)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=8)
    
    # Plot 2: F1 Score Comparison (Bar Chart)
    ax = axes[0, 1]
    f1_no_overlap = [r['f1_score'] for r in results_no_overlap]
    f1_50_overlap = [r['f1_score'] for r in results_50_overlap]
    
    bars1 = ax.bar(x - width/2, f1_no_overlap, width, label='Non-overlapping', 
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, f1_50_overlap, width, label='50% Overlap', 
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Window Size (samples)', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('F1 Score Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(window_sizes)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Accuracy vs F1 Score (Non-overlapping)
    ax = axes[1, 0]
    scatter = ax.scatter(acc_no_overlap, f1_no_overlap, s=300, c=window_sizes,
                        cmap='Blues', alpha=0.6, edgecolors='black', linewidth=2)
    for ws, acc, f1 in zip(window_sizes, acc_no_overlap, f1_no_overlap):
        ax.annotate(f'{ws}', xy=(acc, f1), fontsize=9, ha='center', va='center',
                   fontweight='bold', color='white')
    ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Non-overlapping: Accuracy vs F1', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: 50% Overlap - Accuracy vs F1 Score
    ax = axes[1, 1]
    scatter = ax.scatter(acc_50_overlap, f1_50_overlap, s=300, c=window_sizes,
                        cmap='Reds', alpha=0.6, edgecolors='black', linewidth=2)
    for ws, acc, f1 in zip(window_sizes, acc_50_overlap, f1_50_overlap):
        ax.annotate(f'{ws}', xy=(acc, f1), fontsize=9, ha='center', va='center',
                   fontweight='bold', color='white')
    ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('50% Overlap: Accuracy vs F1', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    
    # Save figure
    output_path = os.path.join('outputs', 'window_accuracy_f1_comparison.png')
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved accuracy & F1 plot to: {output_path}")
    plt.show()

def print_accuracy_summary(results_no_overlap, results_50_overlap):
    """Print summary statistics for accuracy and F1 score."""
    if not results_no_overlap or not results_50_overlap:
        return
    
    print("\n" + "="*70)
    print("ACCURACY & F1 SCORE SUMMARY - NON-OVERLAPPING")
    print("="*70)
    
    accuracies_no = [r['accuracy'] for r in results_no_overlap if r['total'] > 0]
    f1_scores_no = [r['f1_score'] for r in results_no_overlap if r['total'] > 0]
    predictions_no = [r['predictions'] for r in results_no_overlap if r['total'] > 0]
    
    if accuracies_no:
        best_acc_idx = np.argmax(accuracies_no)
        best_f1_idx = np.argmax(f1_scores_no)
        
        print(f"\nAccuracy Statistics:")
        print(f"  Best accuracy: {results_no_overlap[best_acc_idx]['window_size']} "
              f"({max(accuracies_no):.1f}%)")
        print(f"  Average accuracy: {np.mean(accuracies_no):.1f}%")
        print(f"  Std deviation: {np.std(accuracies_no):.1f}%")
        
        print(f"\nF1 Score Statistics:")
        print(f"  Best F1 score: {results_no_overlap[best_f1_idx]['window_size']} "
              f"({max(f1_scores_no):.3f})")
        print(f"  Average F1 score: {np.mean(f1_scores_no):.3f}")
        print(f"  Std deviation: {np.std(f1_scores_no):.3f}")
        
        print(f"\nPrediction Statistics:")
        print(f"  Avg predictions: {np.mean(predictions_no):.1f}")
        print(f"  Std deviation: {np.std(predictions_no):.1f}")
    
    print("\n" + "="*70)
    print("ACCURACY & F1 SCORE SUMMARY - 50% OVERLAP")
    print("="*70)
    
    accuracies_50 = [r['accuracy'] for r in results_50_overlap if r['total'] > 0]
    f1_scores_50 = [r['f1_score'] for r in results_50_overlap if r['total'] > 0]
    predictions_50 = [r['predictions'] for r in results_50_overlap if r['total'] > 0]
    
    if accuracies_50:
        best_acc_idx = np.argmax(accuracies_50)
        best_f1_idx = np.argmax(f1_scores_50)
        
        print(f"\nAccuracy Statistics:")
        print(f"  Best accuracy: {results_50_overlap[best_acc_idx]['window_size']} "
              f"({max(accuracies_50):.1f}%)")
        print(f"  Average accuracy: {np.mean(accuracies_50):.1f}%")
        print(f"  Std deviation: {np.std(accuracies_50):.1f}%")
        
        print(f"\nF1 Score Statistics:")
        print(f"  Best F1 score: {results_50_overlap[best_f1_idx]['window_size']} "
              f"({max(f1_scores_50):.3f})")
        print(f"  Average F1 score: {np.mean(f1_scores_50):.3f}")
        print(f"  Std deviation: {np.std(f1_scores_50):.3f}")
        
        print(f"\nPrediction Statistics:")
        print(f"  Avg predictions: {np.mean(predictions_50):.1f}")
        print(f"  Std deviation: {np.std(predictions_50):.1f}")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    if accuracies_no and accuracies_50 and f1_scores_no and f1_scores_50:
        pred_ratio = np.mean(predictions_50) / (np.mean(predictions_no) + 1e-6)
        acc_diff = np.mean(accuracies_50) - np.mean(accuracies_no)
        f1_diff = np.mean(f1_scores_50) - np.mean(f1_scores_no)
        
        print(f"\nOverlap Effect:")
        print(f"  - 50% overlap generates {pred_ratio:.1f}x more predictions")
        print(f"  - Accuracy difference: {acc_diff:+.1f}%")
        print(f"  - F1 score difference: {f1_diff:+.3f}")
        
        if pred_ratio > 1.5:
            print(f"  - Overlap significantly increases prediction frequency")
        if abs(acc_diff) < 5:
            print(f"  - Accuracy is stable with or without overlap")
        
        # Find sweet spot for F1
        good_results_50 = [r for r in results_50_overlap if r['f1_score'] >= 0.35]
        if good_results_50:
            best_f1 = max(good_results_50, key=lambda x: x['f1_score'])
            print(f"\nBest F1 score (50% overlap): Window size {best_f1['window_size']}")
            print(f"  - F1 Score: {best_f1['f1_score']:.3f}")
            print(f"  - Accuracy: {best_f1['accuracy']:.1f}%")
            print(f"  - Predictions: {best_f1['predictions']}")


def plot_f1_vs_window_size(results_no_overlap, results_50_overlap):
    """
    Plot window size vs F1 score (non-overlapping and 50% overlap).
    Uses the same results produced by run_accuracy_study().
    """
    if not results_no_overlap or not results_50_overlap:
        print("No results to plot for F1 vs window size.")
        return

    window_sizes = [r["window_size"] for r in results_no_overlap]
    f1_no_overlap = [r["f1_score"] for r in results_no_overlap]
    f1_50_overlap = [r["f1_score"] for r in results_50_overlap]

    plt.figure(figsize=(10, 6))

    # Non-overlapping line
    plt.plot(
        window_sizes,
        f1_no_overlap,
        "o-",
        label="Non-overlapping",
        color="#3498db",
        linewidth=2,
        markersize=7,
    )

    # 50% overlap line
    plt.plot(
        window_sizes,
        f1_50_overlap,
        "s-",
        label="50% overlap",
        color="#e74c3c",
        linewidth=2,
        markersize=7,
    )

    plt.xlabel("Window Size (samples)", fontsize=12, fontweight="bold")
    plt.ylabel("F1 Score", fontsize=12, fontweight="bold")
    plt.title("F1 Score vs Window Size", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.ylim(0, 1.0)
    plt.xticks(window_sizes)
    plt.legend(fontsize=11)

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "f1_score_vs_window_size.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved F1 vs window size plot to: {out_path}")
    plt.close()

def main():
    """Run the complete accuracy study."""
    results = run_accuracy_study()
    
    if results is None:
        print("Failed to run accuracy study")
        return
    
    results_no_overlap, results_50_overlap = results
    
    # Print summary
    print_accuracy_summary(results_no_overlap, results_50_overlap)
    
    # Create comparison plots
    print("\nGenerating plots...")
    plot_accuracy_comparison(results_no_overlap, results_50_overlap)

    # Create dedicated F1 vs window size plot
    print("\nGenerating F1 vs window size plot...")
    plot_f1_vs_window_size(results_no_overlap, results_50_overlap)

    # print(f"Seed number: {seed_number}")

if __name__ == "__main__":
    main()