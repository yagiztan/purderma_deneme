from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io

app = Flask(__name__)

# Cihazı belirle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Veri dönüşümleri
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resimleri sabit boyuta getir
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Modeli yükle
model = torch.load("purderma_olan.pth", map_location=device)
model.eval()

# Sınıf isimlerini tanımla (örneğin: ['class1', 'class2', ...])
class_names = ["1.derece Yanık", "2. Melanoma", "3.Akne ve Rosacea", "4.Egzama Hastalığı", "5.Sedef hastalığı resimleri Liken Planus ve ilgili hastalıklar","6.Siğiller Yumuşakçalar ve diğer Viral Enfeksiyonlar"]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Gönderilen resmi oku ve işle
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_image = transform(image).unsqueeze(0).to(device)

        # Model tahmini yap
        with torch.no_grad():
            outputs = model(input_image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

            # En yüksek 5 tahmini al
            top5_prob, top5_indices = torch.topk(probabilities, 5)
            predictions = []
            for i in range(5):
                class_name = class_names[top5_indices[i].item()]
                probability = top5_prob[i].item() * 100  # Yüzdeye çevir
                predictions.append({
                    'class': class_name,
                    'probability': round(probability, 2)
                })

        return jsonify({'predictions': predictions}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)