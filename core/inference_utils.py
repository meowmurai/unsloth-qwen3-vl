import os

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def collect_images(image_dir: str) -> list[str]:
    """Collect all image files from a directory, sorted by name."""
    paths = []
    for entry in sorted(os.listdir(image_dir)):
        if os.path.splitext(entry)[1].lower() in IMAGE_EXTENSIONS:
            paths.append(os.path.join(image_dir, entry))
    return paths
