# create_icons.py - Run this to (re)generate PWA icons from the real logo
#
# Previously this drew a plain "AM" placeholder badge with Pillow's
# ImageDraw - that's what you saw appear instead of the logo when adding
# the site to a phone's home screen. It now resizes the actual brand logo
# (static/img/ASK_ME_Logo.png) down to every size the manifest needs, so
# every icon - including the home-screen one - is the real logo.
import os

from PIL import Image

SOURCE_LOGO = "static/img/ASK_ME_Logo.png"

SIZES = [72, 96, 128, 144, 152, 192, 384, 512]


def generate_icons(source_path: str = SOURCE_LOGO, out_dirs=("static/img/icons", "staticfiles/img/icons")):
    if not os.path.exists(source_path):
        raise FileNotFoundError(
            f"Logo not found at {source_path}. Place the master ASK_ME logo there first."
        )

    with Image.open(source_path) as logo:
        logo = logo.convert("RGBA")
        for size in SIZES:
            # LANCZOS gives the cleanest downscale for sharp edges/text in the logo.
            resized = logo.resize((size, size), Image.LANCZOS)
            filename = f"icon-{size}x{size}.png"
            for out_dir in out_dirs:
                os.makedirs(out_dir, exist_ok=True)
                resized.save(os.path.join(out_dir, filename))
            print(f"Created {filename} ({size}x{size}) from {source_path}")


if __name__ == "__main__":
    print("Generating PWA icons from the real ASK_ME logo...")
    generate_icons()
    print("\u2705 All icons created from the logo!")
