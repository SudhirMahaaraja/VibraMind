from datasets import load_dataset

# Define your custom path
custom_path = r"D:\prec machine\dataaaaa"

# Load the dataset with the cache_dir parameter
ds = load_dataset(
    "DavidNguyen/XJTU-SY_Bearing_Datasets", 
    cache_dir=custom_path
)

print(ds)