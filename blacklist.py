train_sample = [1, 2, 3, 4, 5, 6, 7, 8]
validate_sample = [3, 6, 11, 45, 23]
test_sample = [7, 45, 21, 87, 65, 90]

train_set = set(train_sample)
validate_set = set(validate_sample)
test_set = set(test_sample)

intersect_12 = train_set.intersection(validate_set)
intersect_13 = train_set.intersection(test_set)
intersect_23 = validate_set.intersection(test_set)

print(intersect_12, intersect_13, intersect_23)

train_set = train_set - intersect_12
train_set = train_set - intersect_13
validate_set = validate_set - intersect_23




