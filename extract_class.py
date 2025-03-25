import scipy.io

meta = scipy.io.loadmat('Dataset/cars_annos.mat')
class_names = meta['class_names'][0]

# Convert to Python list
names = [name[0] for name in class_names]

# Print as YAML array
print("names: [")
for i, name in enumerate(names):
    comma = ',' if i < len(names) - 1 else ''
    print(f'  "{name}"{comma}')
print("]")
