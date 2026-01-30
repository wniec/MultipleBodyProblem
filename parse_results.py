import os
import re
import csv


TARGET_DIR = "results"
OUTPUT_FILE = "benchmarks.csv"


def extract_data():
    filename_pattern = re.compile(r"raport_(\d+)_(\d+)\.txt")

    fps_pattern = re.compile(r"Mean FPS:\s*([\d\.]+)")

    extracted_data = []

    if not os.path.exists(TARGET_DIR):
        print(f"Error: Directory '{TARGET_DIR}' not found.")
        return

    print(f"Scanning directory: {TARGET_DIR}...")


    for filename in os.listdir(TARGET_DIR):
        name_match = filename_pattern.match(filename)
        if name_match:
            n_bodies = name_match.group(1)
            tpb = name_match.group(2)

            filepath = os.path.join(TARGET_DIR, filename)

            try:
                with open(filepath, 'r') as f:
                    content = f.read()

                    fps_match = fps_pattern.search(content)
                    if fps_match:
                        fps = fps_match.group(1)
                        extracted_data.append({
                            "N": int(n_bodies),
                            "TPB": int(tpb),
                            "FPS": float(fps)
                        })
                    else:
                        print(f"Warning: 'Mean FPS' not found in {filename}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    extracted_data.sort(key=lambda x: (x["N"], x["TPB"]))

    if extracted_data:
        with open(OUTPUT_FILE, 'w', newline='') as csvfile:
            fieldnames = ['N', 'TPB', 'FPS']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in extracted_data:
                writer.writerow(row)

        print(f"Success! Processed {len(extracted_data)} files.")
        print(f"Data saved to: {OUTPUT_FILE}")
    else:
        print("No matching files found.")


if __name__ == "__main__":
    extract_data()