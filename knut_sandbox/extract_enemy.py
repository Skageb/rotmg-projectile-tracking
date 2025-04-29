from PIL import Image
import os
import re

def extract_blocks_from_image(image_path, output_dir, block_width, block_height):
    """Extracts non-transparent blocks from a single image."""
    img = Image.open(image_path)
    width, height = img.size

    # Ensure we're working with RGBA to check transparency
    img = img.convert("RGBA")
    
    count = 0

    # Process all blocks
    for y in range(0, height, block_height):
        for x in range(0, width, block_width):
            # Check if we're within image bounds
            if x + block_width > width or y + block_height > height:
                continue

            # Extract the block
            box = (x, y, x + block_width, y + block_height)
            block = img.crop(box)

            # Check transparency
            alpha = block.getchannel("A")
            if all(pixel == 0 for pixel in alpha.getdata()):
                continue  # Skip fully transparent blocks

            # Save the block
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            block.save(os.path.join(output_dir, f"{base_name}_{count}.png"))
            count += 1

    return count

def process_folder(input_folder="projectlie", output_folder="projectile_blocks"):
    """Processes all PNG images in the input folder."""
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Regex to extract block size from filename (e.g., "chars8x8" -> (8, 8))
    # Added re.IGNORECASE to handle different casings
    size_pattern = re.compile(r"(\d+)x(\d+)", re.IGNORECASE)

    # Process all PNG files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            print(f"Processing {filename}...")
            # Extract block size from filename
            match = size_pattern.search(filename)
            if not match:
                print(f"Skipping {filename}: Could not determine block size from filename.")
                continue

            block_width = int(match.group(1))
            block_height = int(match.group(2))

            # Process the image
            image_path = os.path.join(input_folder, filename)
            print(f"Processing {filename} with block size {block_width}x{block_height}...")
            num_blocks = extract_blocks_from_image(image_path, output_folder, block_width, block_height)
            print(f"Extracted {num_blocks} blocks from {filename}.")

    print(f"All images processed. Output saved to '{output_folder}'.")


process_folder()