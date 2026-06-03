import neural_network

min_test_num = 500
max_test_num = 90000
correct = 0
for i in range(min_test_num, max_test_num + 1):
    label, prediction = neural_network.classify(i)
    correct += (i % 2 == 0 and label == "even") or (i % 2 != 0 and label == "odd")

print(f"{100 * correct / (max_test_num - min_test_num + 1)}% accuracy!")