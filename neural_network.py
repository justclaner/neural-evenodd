import generate_training
import math
import random
import json
import time

NUM_BITS = generate_training.NUM_BITS

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

weights = [random.uniform(-1, 1) for _ in range(NUM_BITS)]
bias = random.uniform(-1, 1)

def save_model(filename = "model.json"):
    with open(filename, "w") as f:
        json.dump({"weights": weights, "bias": bias}, f)

def load_model(filename = "model.json"):
    global weights, bias
    with open(filename, "r") as f:
        data = json.load(f)
    weights = data["weights"]
    bias = data["bias"]

# truncates from the left
def number_to_bits(number, num_bits):
    binary_str = bin(abs(number))[2:]
    binary_str = binary_str[-num_bits:].zfill(num_bits)
    return [int(b) for b in binary_str]

def predict(bits):
    z = sum(w * x for w, x in zip(weights, bits)) + bias
    return sigmoid(z)

def train(trials):
    global weights, bias
    if trials < 1:
        return
    
    training_data = []
    with open(generate_training.file_name, "r") as f:
        for line in f:
            binary_str, label = line.strip().split("\t")
            training_data.append((binary_str, int(label)))
    
    for _ in range(trials):
        random.shuffle(training_data)
        for binary_str, target in training_data:
            bits = number_to_bits(int(binary_str, 2), NUM_BITS)
            prediction = predict(bits)
            error = target - prediction
            for i in range(NUM_BITS):
                weights[i] += 0.1 * error * bits[i]
            bias += 0.1 * error

def classify(number):
    bits = number_to_bits(number, NUM_BITS)
    prediction = predict(bits)
    return ("odd" if prediction >= 0.5 else "even", prediction)

try:
    load_model()
    print("Loaded existing model.")
except FileNotFoundError:
    print("Creating new model.")
    
if __name__ == "__main__":
    
    start_time = time.perf_counter()
    print("Starting training...")
    train(100)
    print(f"Training took {time.perf_counter() - start_time}s!")

    save_model()