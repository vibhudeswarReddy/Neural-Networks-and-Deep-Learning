!pip install tensorflow pillow numpy
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
# Check environment: Colab vs local
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
# Load pre-trained ResNet50 model
model = ResNet50(weights="imagenet")
def identify_species(img_path, top_n=5):
    """Predict species from an image and display top N predictions"""
    try:
        img = image.load_img(img_path, target_size=(224, 224))
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    preds = model.predict(img_array)
    decoded = decode_predictions(preds, top=top_n)[0]
    print(f"\nTop {top_n} predictions for {img_path}:\n")
    found = False
    for _, label, confidence in decoded:
        label_clean = label.lower().replace("_", " ")
        print(f"Prediction: {label_clean} ({confidence:.2f})")
        # Check if it's dog, cat, or bird
        if any(word in label_clean for word in ["dog", "shepherd", "retriever", "terrier",
                                                "bulldog", "poodle", "husky", "rottweiler",
                                                "doberman", "pitbull"]):
            print(f"\n🐶 Detected Dog breed: {label_clean} ({confidence:.2f})\n")
            found = True
            break
        if any(word in label_clean for word in ["cat", "siamese", "persian", "tabby", "egyptian mau"]):
            print(f"\n🐱 Detected Cat species: {label_clean} ({confidence:.2f})\n")
            found = True
            break
        if any(word in label_clean for word in ["bird", "parrot", "macaw", "cockatoo", "sparrow",
                                                "hen", "peacock", "eagle", "owl", "duck",
                                                "goose", "canary", "goldfinch", "hummingbird",
                                                "flamingo", "robin", "pelican", "kingfisher",
                                                "penguin"]):
            print(f"\n🐦 Detected Bird species: {label_clean} ({confidence:.2f})\n")
            found = True
            break
    if not found:
        print("\n❓ Unknown species (not dog/cat/bird)\n")
# HEADLESS FILE SELECTION
if IN_COLAB:
    print("📁 Please select an image file to upload.")
    uploaded = files.upload()  # Colab file chooser
    for filename in uploaded.keys():
        print(f"\nProcessing: {filename}")
        identify_species(filename)
else:
    img_path = input("📁 Enter image path (drag or paste the file here): ").strip()
    identify_species(img_path)
