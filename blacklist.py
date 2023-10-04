with open('files/accuracy_history_small.txt', 'r') as file:
    lines = file.readlines()
print(lines)
for i in range(len(lines)):
    line = eval(lines[i])
    print(line)