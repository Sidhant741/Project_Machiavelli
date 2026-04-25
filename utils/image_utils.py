import base64
from io import BytesIO
from typing import Optional

try:
    from PIL import Image
except ImportError:
    Image = None

def encode_image_to_base64(image_path: str) -> str:
    """
    Read an image file from the given path and encode it as a base64 string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def decode_base64_to_image(base64_string: str) -> Optional["Image.Image"]:
    """
    Decode a base64 string and return a PIL Image object.
    Requires Pillow to be installed.
    """
    if Image is None:
        raise ImportError("Pillow (PIL) is not installed. Run `pip install Pillow` to use this function.")
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image

def decode_base64_to_bytes(base64_string: str) -> bytes:
    """
    Decode a base64 string and return the raw bytes.
    Useful if you don't want to use PIL and just need to save or stream the file.
    """
    return base64.b64decode(base64_string)

def save_base64_image(base64_string: str, output_path: str) -> None:
    """
    Decode a base64 string and save it directly to a file path.
    """
    with open(output_path, "wb") as out_file:
        out_file.write(base64.b64decode(base64_string))
