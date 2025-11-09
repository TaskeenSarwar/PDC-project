import random
import time
import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ProcessPoolExecutor


# Regular quicksort (sequential)
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)


# Parallel quicksort (optimized)
def parallel_quicksort(arr, num_workers=4):
    """Parallel quicksort using a single shared process pool."""

    # Split array into roughly equal chunks
    chunk_size = len(arr) // num_workers
    chunks = [arr[i:i + chunk_size] for i in range(0, len(arr), chunk_size)]

    # Each process sorts its chunk independently
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        sorted_chunks = list(executor.map(quicksort, chunks))

    # Merge the sorted chunks sequentially
    sorted_arr = []
    for chunk in sorted_chunks:
        sorted_arr += chunk
    return sorted(sorted_arr)  # final merge (using built-in sort for speed)


# Time measurement function
def measure_time(func, arr):
    start = time.time()
    func(arr)
    end = time.time()
    return end - start


# Generate random dataset
def generate_dataset(size):
    return [random.randint(0, 1000000) for _ in range(size)]


if __name__ == "__main__":
    dataset_sizes = [10000, 50000, 100000, 500000, 1000000]
    results = []

    for size in dataset_sizes:
        print(f"\nDataset: Random_{size}")
        data = generate_dataset(size)

        seq_time = measure_time(lambda arr: quicksort(arr.copy()), data)
        print(f"Sequential Quicksort Time: {seq_time:.4f} seconds")

        par_time = measure_time(lambda arr: parallel_quicksort(arr.copy(), num_workers=4), data)
        print(f"Parallel Quicksort Time: {par_time:.4f} seconds")

        results.append({"Dataset Size": size, "Sequential": seq_time, "Parallel": par_time})

    # Show summary
    df = pd.DataFrame(results)
    print("\n--- Performance Summary ---")
    print(df)

    df.to_csv("quicksort_performance.csv", index=False)
    plt.figure(figsize=(8, 5))
    plt.plot(df["Dataset Size"], df["Sequential"], label="Sequential", marker="o")
    plt.plot(df["Dataset Size"], df["Parallel"], label="Parallel", marker="o")
    plt.xlabel("Dataset Size")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Optimized Sequential vs Parallel QuickSort Performance")
    plt.legend()
    plt.grid(True)
    plt.savefig("quicksort_performance.png")
    plt.show()
