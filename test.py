import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pastequeflow as pf
from Constants import Constants as Cst


cst = Cst()

csv_loader = pf.data.datasources.CSVLoader(
    train_val_csv_path=cst.file.train_val_csv_path,
    test_csv_path=cst.file.test_csv_path,
    x_col="image",
    y_col="char"
)

print(f"Classes ({len(csv_loader.classes)}):\n", csv_loader.classes)
print("\n\n")
print("Class mappings:\n", csv_loader.class_mappings)
print("\n\n")
print("Classes repartition:\n", csv_loader.classes_repartition)
print("\n\n")
print("Weights:\n", csv_loader.weights)
# print("\n\n")

# print(csv_loader.get_train_val_data())
# print("\n\n")
# print(csv_loader.get_testing_data())

print("\n\n--------------------------------------------------------------\n\n")

dataset_builder = pf.data.dataset_builders.ImageDatasetBuilder(
    train_img_dir=cst.file.train_val_data_dir,
    test_img_dir=cst.file.test_data_dir
)

train_ds, val_ds = dataset_builder.get_train_val_datasets(x_y_data=csv_loader.get_train_val_data(), classes=csv_loader.classes)
test_ds = dataset_builder.get_testing_dataset(x_y_data=csv_loader.get_testing_data(), classes=csv_loader.classes)

print("train dataset:")
for image, label in train_ds.take(1):
    print("Image shape:", image.shape)
    print("Label:", label)

print("\nval dataset:")
for image, label in val_ds.take(1):
    print("Image shape:", image.shape)
    print("Label:", label)

print("\ntest dataset:")
for image in test_ds.take(1):
    print("Image shape:", image.shape)


