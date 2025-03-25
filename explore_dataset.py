import scipy.io
from pprint import pprint

# Load fail annotation
mat = scipy.io.loadmat('Dataset/cars_annos.mat')
annotations = mat['annotations'][0]  # annotations adalah array

# Tengok satu contoh data gambar pertama
sample = annotations[0]

info = {
    "Filename": sample[0][0],
    "Bounding Box": {
        "x1": sample[1][0][0],
        "y1": sample[2][0][0],
        "x2": sample[3][0][0],
        "y2": sample[4][0][0]
    },
    "Class ID": sample[5][0][0],
    "Is Test Image": bool(sample[6][0][0])
}

print("Contoh satu gambar:")
pprint(info)

# Kira jumlah class
class_ids = set()
for anno in annotations:
    class_ids.add(anno[5][0][0])

print(f"\nJumlah class unik: {len(class_ids)}")
