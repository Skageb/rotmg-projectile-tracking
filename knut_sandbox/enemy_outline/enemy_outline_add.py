import os
from PIL import Image, ImageChops, ImageFilter


def add_outline(img, thickness=1, outline_color=(0, 0, 0, 255)):
    """
    Adds a crisp black outline to a transparent image by dilating the alpha channel.
    Uses an expansion approach to ensure the outline sits around the enemy.

    Parameters:
      img: PIL.Image object in RGBA mode.
      thickness: outline thickness in pixels.
      outline_color: RGBA tuple for the outline color.

    Returns:
      A new PIL.Image with an outline added.
    """
    if thickness <= 0:
        return img

    # Ensure the image is in RGBA.
    img = img.convert("RGBA")

    # Create a new blank image to accommodate the outline
    new_size = (img.width + thickness * 2, img.height + thickness * 2)
    canvas = Image.new("RGBA", new_size, (0, 0, 0, 0))
    canvas.paste(img, (thickness, thickness))

    # Extract and dilate the alpha channel
    alpha = canvas.split()[-1]
    dilated = alpha.filter(ImageFilter.MaxFilter(thickness * 2 + 1))

    # The outline mask is the dilated alpha minus the original alpha
    outline_mask = ImageChops.subtract(dilated, alpha)

    # Create an image for the outline using the outline_color
    outline_img = Image.new("RGBA", new_size, (0, 0, 0, 0))
    outline_img.paste(outline_color, mask=outline_mask)

    # Composite the outline and the original image
    final_img = Image.alpha_composite(outline_img, canvas)
    return final_img


enemy_img = Image.open("knut_sandbox/enemy_outline/chars8x8rEncounters_58.png")

# Resize the enemy image 4x its original dimensions
enemy_img_resized = enemy_img.resize(
    (enemy_img.width * 2, enemy_img.height * 2), resample=Image.BICUBIC)
enemy_img_resized = add_outline(enemy_img_resized, thickness=1)  # Add a 1-pixel black outline  # Add a 1-pixel black outline
enemy_img_resized = enemy_img_resized.resize(
    (enemy_img_resized.width * 2, enemy_img_resized.height * 2), resample=Image.BOX)
e_width, e_height = enemy_img_resized.size
enemy_img_resized = add_outline(enemy_img_resized, thickness=1)  # Add a 1-pixel black outline  # Add a 1-pixel black outline

# Display enemy image
enemy_img_resized.show()