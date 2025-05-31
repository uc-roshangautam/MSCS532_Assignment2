import random
import time
import sys
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np

def merge_sort(arr: List[int]) -> List[int]:
    """
    Implements merge sort algorithm.
    
    Args:
        arr: List of integers to sort
        
    Returns:
        Sorted list of integers
    """
    if len(arr) <= 1:
        return arr.copy()
    
    # Divide
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Conquer (merge)
    return merge(left, right)

def merge(left: List[int], right: List[int]) -> List[int]:
    """
    Merges two sorted lists into one sorted list.
    
    Args:
        left: First sorted list
        right: Second sorted list
        
    Returns:
        Merged sorted list
    """
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Add remaining elements
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result

def quick_sort(arr: List[int]) -> List[int]:
    """
    Implements quick sort algorithm using random pivot selection.
    
    Args:
        arr: List of integers to sort
        
    Returns:
        Sorted list of integers
    """
    if len(arr) <= 1:
        return arr.copy()
    
    result = arr.copy()
    _quick_sort_inplace(result, 0, len(result) - 1)
    return result

def _quick_sort_inplace(arr: List[int], low: int, high: int) -> None:
    """
    In-place quick sort implementation.
    
    Args:
        arr: Array to sort
        low: Starting index
        high: Ending index
    """
    if low < high:
        # Random pivot selection to avoid worst case
        pivot_idx = random.randint(low, high)
        arr[pivot_idx], arr[high] = arr[high], arr[pivot_idx]
        
        pi = partition(arr, low, high)
        _quick_sort_inplace(arr, low, pi - 1)
        _quick_sort_inplace(arr, pi + 1, high)

def partition(arr: List[int], low: int, high: int) -> int:
    """
    Partitions array around pivot element.
    
    Args:
        arr: Array to partition
        low: Starting index
        high: Ending index (pivot location)
        
    Returns:
        Final position of pivot
    """
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

# Performance testing functions
def generate_test_data(size: int, data_type: str) -> List[int]:
    """
    Generates test data of specified type and size.
    
    Args:
        size: Number of elements
        data_type: Type of data ('random', 'sorted', 'reverse')
        
    Returns:
        List of test data
    """
    if data_type == 'random':
        return [random.randint(1, size * 10) for _ in range(size)]
    elif data_type == 'sorted':
        return list(range(1, size + 1))
    elif data_type == 'reverse':
        return list(range(size, 0, -1))
    else:
        raise ValueError("Invalid data type")

def measure_performance(sort_func, data: List[int]) -> Tuple[float, List[int]]:
    """
    Measures execution time and memory usage of sorting function.
    
    Args:
        sort_func: Sorting function to test
        data: Input data
        
    Returns:
        Tuple of (execution_time, sorted_result)
    """
    start_time = time.time()
    result = sort_func(data)
    end_time = time.time()
    
    return end_time - start_time, result

def run_performance_tests():
    """
    Runs comprehensive performance tests on both algorithms.
    """
    sizes = [100, 500, 1000, 2500, 5000, 10000]
    data_types = ['random', 'sorted', 'reverse']
    
    results = {
        'merge_sort': {'random': [], 'sorted': [], 'reverse': []},
        'quick_sort': {'random': [], 'sorted': [], 'reverse': []}
    }
    
    print("Performance Testing Results")
    print("=" * 50)
    
    for size in sizes:
        print(f"\nTesting with {size} elements:")
        
        for data_type in data_types:
            # Generate test data
            test_data = generate_test_data(size, data_type)
            
            # Run multiple trials for more accurate timing
            merge_times = []
            quick_times = []
            
            for trial in range(3):  # 3 trials for averaging
                # Test Merge Sort
                merge_time, merge_result = measure_performance(merge_sort, test_data)
                merge_times.append(merge_time)
                
                # Test Quick Sort
                quick_time, quick_result = measure_performance(quick_sort, test_data)
                quick_times.append(quick_time)
                
                # Verify correctness on first trial
                if trial == 0:
                    assert merge_result == quick_result == sorted(test_data)
            
            # Average the results
            avg_merge_time = sum(merge_times) / len(merge_times)
            avg_quick_time = sum(quick_times) / len(quick_times)
            
            results['merge_sort'][data_type].append((size, avg_merge_time))
            results['quick_sort'][data_type].append((size, avg_quick_time))
            
            print(f"  {data_type.capitalize()} data:")
            print(f"    Merge Sort: {avg_merge_time:.6f}s")
            print(f"    Quick Sort: {avg_quick_time:.6f}s")
            print(f"    Ratio (Q/M): {avg_quick_time/avg_merge_time:.2f}")
    
    return results

def create_performance_tables(results):
    """
    Creates formatted performance tables for the report.
    """
    sizes = [100, 500, 1000, 2500, 5000, 10000]
    data_types = ['random', 'sorted', 'reverse']
    
    print("\n" + "="*80)
    print("PERFORMANCE TABLES")
    print("="*80)
    
    for data_type in data_types:
        print(f"\n{data_type.upper()} DATA PERFORMANCE (seconds)")
        print("-" * 60)
        print(f"{'Size':<8} {'Merge Sort':<12} {'Quick Sort':<12} {'Ratio (Q/M)':<12} {'Winner':<10}")
        print("-" * 60)
        
        merge_data = results['merge_sort'][data_type]
        quick_data = results['quick_sort'][data_type]
        
        for i, size in enumerate(sizes):
            merge_time = merge_data[i][1]
            quick_time = quick_data[i][1]
            ratio = quick_time / merge_time
            winner = "Quick" if quick_time < merge_time else "Merge"
            
            print(f"{size:<8} {merge_time:<12.6f} {quick_time:<12.6f} {ratio:<12.2f} {winner:<10}")

def plot_performance_graphs(results):
    """
    Creates comprehensive performance visualization graphs.
    """
    sizes = [100, 500, 1000, 2500, 5000, 10000]
    data_types = ['random', 'sorted', 'reverse']
    
    # Create subplot figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Merge Sort vs Quick Sort Performance Analysis', fontsize=16, fontweight='bold')
    
    # Performance comparison plots
    for i, data_type in enumerate(data_types):
        merge_data = results['merge_sort'][data_type]
        quick_data = results['quick_sort'][data_type]
        
        merge_times = [x[1] for x in merge_data]
        quick_times = [x[1] for x in quick_data]
        
        # Linear scale plot
        axes[0, i].plot(sizes, merge_times, 'b-o', linewidth=2, markersize=6, label='Merge Sort')
        axes[0, i].plot(sizes, quick_times, 'r-s', linewidth=2, markersize=6, label='Quick Sort')
        axes[0, i].set_title(f'{data_type.capitalize()} Data - Linear Scale', fontweight='bold')
        axes[0, i].set_xlabel('Array Size')
        axes[0, i].set_ylabel('Time (seconds)')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        
        # Log scale plot
        axes[1, i].loglog(sizes, merge_times, 'b-o', linewidth=2, markersize=6, label='Merge Sort')
        axes[1, i].loglog(sizes, quick_times, 'r-s', linewidth=2, markersize=6, label='Quick Sort')
        axes[1, i].set_title(f'{data_type.capitalize()} Data - Log Scale', fontweight='bold')
        axes[1, i].set_xlabel('Array Size')
        axes[1, i].set_ylabel('Time (seconds)')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
        
        # Add theoretical complexity lines
        if i == 0:  # Only on first plot to avoid clutter
            theoretical_nlogn = [size * np.log2(size) * 1e-6 for size in sizes]
            axes[1, i].loglog(sizes, theoretical_nlogn, 'g--', alpha=0.7, label='O(n log n)')
            axes[1, i].legend()
    
    plt.tight_layout()
    plt.show()

def create_ratio_analysis(results):
    """
    Creates performance ratio analysis table and graph.
    """
    sizes = [100, 500, 1000, 2500, 5000, 10000]
    data_types = ['random', 'sorted', 'reverse']
    
    print(f"\n{'PERFORMANCE RATIO ANALYSIS (Quick Sort / Merge Sort)'}")
    print("-" * 70)
    print(f"{'Size':<8}", end="")
    for data_type in data_types:
        print(f"{data_type.capitalize():<12}", end="")
    print(f"{'Average':<12}")
    print("-" * 70)
    
    # Calculate and display ratios
    for i, size in enumerate(sizes):
        print(f"{size:<8}", end="")
        ratios = []
        
        for data_type in data_types:
            merge_time = results['merge_sort'][data_type][i][1]
            quick_time = results['quick_sort'][data_type][i][1]
            ratio = quick_time / merge_time
            ratios.append(ratio)
            print(f"{ratio:<12.2f}", end="")
        
        avg_ratio = sum(ratios) / len(ratios)
        print(f"{avg_ratio:<12.2f}")
    
    # Create ratio visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Ratio trends by data type
    for data_type in data_types:
        ratios = []
        for i in range(len(sizes)):
            merge_time = results['merge_sort'][data_type][i][1]
            quick_time = results['quick_sort'][data_type][i][1]
            ratio = quick_time / merge_time
            ratios.append(ratio)
        
        ax1.plot(sizes, ratios, 'o-', linewidth=2, markersize=6, label=f'{data_type.capitalize()}')
    
    ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Equal Performance')
    ax1.set_title('Performance Ratio Trends\n(Quick Sort / Merge Sort)', fontweight='bold')
    ax1.set_xlabel('Array Size')
    ax1.set_ylabel('Time Ratio')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Average ratio by size
    avg_ratios = []
    for i in range(len(sizes)):
        ratios = []
        for data_type in data_types:
            merge_time = results['merge_sort'][data_type][i][1]
            quick_time = results['quick_sort'][data_type][i][1]
            ratio = quick_time / merge_time
            ratios.append(ratio)
        avg_ratios.append(sum(ratios) / len(ratios))
    
    bars = ax2.bar(range(len(sizes)), avg_ratios, alpha=0.7, color='skyblue', edgecolor='navy')
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal Performance')
    ax2.set_title('Average Performance Ratio by Size', fontweight='bold')
    ax2.set_xlabel('Array Size')
    ax2.set_ylabel('Average Time Ratio')
    ax2.set_xticks(range(len(sizes)))
    ax2.set_xticklabels(sizes)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{avg_ratios[i]:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def analyze_complexity_growth(results):
    """
    Analyzes how actual performance compares to theoretical complexity.
    """
    sizes = [100, 500, 1000, 2500, 5000, 10000]
    
    print(f"\n{'COMPLEXITY GROWTH ANALYSIS'}")
    print("=" * 50)
    
    # Calculate growth ratios
    for algorithm in ['merge_sort', 'quick_sort']:
        print(f"\n{algorithm.replace('_', ' ').title()} Growth Analysis:")
        print("-" * 40)
        print(f"{'Size Ratio':<12} {'Time Ratio':<15} {'Expected (n log n)':<20} {'Efficiency':<12}")
        print("-" * 40)
        
        # Use random data for this analysis
        times = [x[1] for x in results[algorithm]['random']]
        
        for i in range(1, len(sizes)):
            size_ratio = sizes[i] / sizes[i-1]
            time_ratio = times[i] / times[i-1]
            
            # Expected ratio for O(n log n)
            expected_ratio = (sizes[i] * np.log2(sizes[i])) / (sizes[i-1] * np.log2(sizes[i-1]))
            
            efficiency = expected_ratio / time_ratio
            
            print(f"{size_ratio:<12.1f} {time_ratio:<15.2f} {expected_ratio:<20.2f} {efficiency:<12.2f}")

# Memory usage analysis function
def estimate_memory_usage():
    """
    Provides theoretical memory usage analysis.
    """
    print(f"\n{'MEMORY USAGE ANALYSIS'}")
    print("=" * 50)
    print(f"{'Algorithm':<15} {'Space Complexity':<20} {'For 10,000 elements':<20} {'Notes'}")
    print("-" * 80)
    print(f"{'Merge Sort':<15} {'O(n)':<20} {'~40 KB additional':<20} {'Temporary arrays'}")
    print(f"{'Quick Sort':<15} {'O(log n) avg':<20} {'~52 bytes avg':<20} {'Recursion stack'}")
    print(f"{'Quick Sort':<15} {'O(n) worst':<20} {'~40 KB worst':<20} {'Deep recursion'}")
    
    # Create memory usage visualization
    sizes = [100, 500, 1000, 2500, 5000, 10000]
    
    merge_memory = [size * 4 for size in sizes]  # 4 bytes per int, O(n) additional
    quick_memory_avg = [np.log2(size) * 8 for size in sizes]  # 8 bytes per stack frame, O(log n)
    quick_memory_worst = [size * 8 for size in sizes]  # O(n) in worst case
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    ax.plot(sizes, merge_memory, 'b-o', linewidth=2, markersize=6, label='Merge Sort O(n)')
    ax.plot(sizes, quick_memory_avg, 'g-s', linewidth=2, markersize=6, label='Quick Sort O(log n) avg')
    ax.plot(sizes, quick_memory_worst, 'r--^', linewidth=2, markersize=6, label='Quick Sort O(n) worst')
    
    ax.set_title('Memory Usage Comparison', fontweight='bold', fontsize=14)
    ax.set_xlabel('Array Size', fontsize=12)
    ax.set_ylabel('Additional Memory (bytes)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.show()

# Example usage and testing
if __name__ == "__main__":
    # Set recursion limit for large datasets
    sys.setrecursionlimit(10000)
    
    # Run comprehensive performance tests
    performance_results = run_performance_tests()
    
    # Generate performance tables
    create_performance_tables(performance_results)
    
    # Create performance ratio analysis
    create_ratio_analysis(performance_results)
    
    # Analyze complexity growth
    analyze_complexity_growth(performance_results)
    
    # Memory usage analysis
    estimate_memory_usage()
    
    # Generate performance graphs
    plot_performance_graphs(performance_results)
    
    # Example with small dataset
    print("\nExample with small dataset:")
    test_arr = [64, 34, 25, 12, 22, 11, 90]
    print(f"Original: {test_arr}")
    print(f"Merge Sort: {merge_sort(test_arr)}")
    print(f"Quick Sort: {quick_sort(test_arr)}")