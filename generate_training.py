import math
import time

min_num = -100000
max_num = 100000
file_name = "training1.txt"
NUM_BITS = math.ceil(math.log2(max_num - min_num + 1))

start_time = time.perf_counter()
print("Started creating training data...")

output = ""
for i in range(min_num, max_num + 1):
    binary_str = bin(abs(i))[2:] 
    label = 1 if i % 2 != 0 else 0
    output += f"{binary_str}\t{label}\n"
print(f"Finished creating training data after {time.perf_counter() - start_time}s!")

start_time = time.perf_counter()
print("Started writing training data to file...")
with open(file_name, "w") as f:
    f.write(output)
print(f"Finished writing training data to file after {time.perf_counter() - start_time}s!")