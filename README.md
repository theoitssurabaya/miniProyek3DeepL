# ATM Python Eye (OWL-ViT Local Demo)

Strong Python script using massive OWL-ViT brain for tribe ATM security. This project uses the device camera to directly detect faces, masks, hats, and sunglasses without any web browser.

## How Brain Works (Logic)
Instead of relying on web frameworks and smaller heuristic models, we use a single, huge zero-shot vision transformer:

* **OWL-ViT (`google/owlvit-base-patch32`)**: A zero-shot text-conditioned object detection model from Hugging Face Transformers.

**The "Why"**:
We just prompt the brain with texts like "a face", "a medical mask", "a hat", "sunglasses". The brain understands these concepts without explicit custom training. Running it natively in Python allows us to utilize PC hardware directly, completely removing the browser. However, because the brain is massive, if the PC does not have a strong GPU, the framerate will be extremely slow.

## Installation & Setup

1. Open terminal in the project directory.
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Activate the virtual environment if not already active (`source venv/bin/activate`).
2. Run the script:
   ```bash
   python owl_demo.py
   ```
3. An OpenCV window will open showing the camera feed.
4. Put on a mask, hat, or sunglasses to trigger the DANGER alert on screen.
