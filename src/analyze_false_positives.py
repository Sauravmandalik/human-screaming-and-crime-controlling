
import csv
from collections import Counter

CSV_PATH = "external_datasets/esc50.csv"

false_positives = [
    "1-30039-A-26.wav",
    "1-31482-B-42.wav",
    "2-43802-A-42.wav",
    "2-50666-A-20.wav",
    "2-70052-B-42.wav",
    "2-87795-A-24.wav",
    "3-126113-A-26.wav",
    "3-151081-A-20.wav",
    "3-151081-B-20.wav",
    "3-51909-B-42.wav",
    "4-102871-A-42.wav",
    "4-121532-A-42.wav",
    "4-132816-A-26.wav",
    "4-155670-A-26.wav",
    "4-164243-A-26.wav",
    "4-181599-A-26.wav",
    "4-189830-A-22.wav",
    "4-67358-A-42.wav",
    "4-80761-A-42.wav",
    "4-90014-B-42.wav",
    "5-117118-A-42.wav",
    "5-117122-A-42.wav",
    "5-150409-A-42.wav",
    "5-184323-A-42.wav",
    "NBMB_12_3_0_58_0_0_0.wav",
    "laughing_20261107_141838.wav",
]

# Read the official ESC-50 metadata
metadata = {}

with open(CSV_PATH, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        metadata[row["filename"]] = row["category"]

# Match false positives with their sound categories
category_counts = Counter()

print("\nFALSE POSITIVE SOUND ANALYSIS\n")

for filename in false_positives:
    category = metadata.get(filename, "Not found in ESC-50")

    print(f"{filename}  -->  {category}")

    if category != "Not found in ESC-50":
        category_counts[category] += 1

print("\nCATEGORY SUMMARY\n")

for category, count in category_counts.most_common():
    print(f"{category}: {count}")

print("\nAnalysis complete. No dataset files were changed.")
