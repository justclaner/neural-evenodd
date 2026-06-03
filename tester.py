import neural_network

min_test_num = -3094
max_test_num = 821300
correct = 0

max_incorrect_displayed = 5
incorrect = 0
for i in range(min_test_num, max_test_num + 1):
    label, prediction = neural_network.classify(i)
    prediction_correct = (i % 2 == 0 and label == "even") or (i % 2 != 0 and label == "odd")
    correct += prediction_correct
    if not prediction_correct and incorrect < max_incorrect_displayed:
        print(f"{i} was predicted incorrectly to be {label}!")
        incorrect += 1

print(f"{100 * correct / (max_test_num - min_test_num + 1)}% accuracy!")