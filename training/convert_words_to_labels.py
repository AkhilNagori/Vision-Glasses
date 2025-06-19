import os

def convert_words_to_labels(words_txt_path, output_labels_path, images_base_path):
    """
    Converts words.txt from the IAM dataset into a labels.txt file for OCR training.

    Args:
        words_txt_path (str): Path to words.txt file.
        output_labels_path (str): Path to save labels.txt.
        images_base_path (str): Base directory where line images are stored.
    """
    if not os.path.exists(words_txt_path):
        print(f"Error: File not found -> {words_txt_path}")
        return

    lines_converted = []

    with open(words_txt_path, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue  # Skip comments and empty lines

            parts = line.strip().split()
            if len(parts) < 9:
                continue  # Skip incomplete entries

            word_id = parts[0]  # e.g., a01-000u-00-00
            status = parts[1]   # ok or err
            transcription = parts[-1]  # last element is the word

            if status != "ok":
                continue  # skip poor-quality segmentations

            folder = word_id[:3]          # e.g., a01
            subfolder = word_id[:7]       # e.g., a01-000u
            filename = word_id + ".png"   # full image filename

            image_path = os.path.join(images_base_path, folder, subfolder, filename)

            # Some image files may be missing; optionally check existence
            # if not os.path.exists(image_path):
            #     continue

            lines_converted.append(f"{image_path}\t{transcription}")

    # Save to labels.txt
    with open(output_labels_path, "w") as out:
        out.write("\n".join(lines_converted))

    print(f"✅ Done! Saved {len(lines_converted)} entries to '{output_labels_path}'")


# -----------------------------
# 📌 Run this script directly:
# -----------------------------
if __name__ == "__main__":
    # Set these paths based on your folder structure
    words_txt = "data/words.txt"
    output_labels = "data/labels.txt"
    image_root = "data/lines"

    convert_words_to_labels(words_txt, output_labels, image_root)
