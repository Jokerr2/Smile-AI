from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import numpy as np
import base64
from io import BytesIO
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

app = Flask(__name__, static_folder='.', static_url_path='')
# Configure CORS to allow requests from your frontend
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5000", "http://127.0.0.1:5000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Create database tables
with app.app_context():
    db.create_all()

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model paths
CLASSIFICATION_MODEL_PATH = "classification_model.h5"
SEGMENTATION_MODELS = {
    "Mandibular Left First Molar": "mandibular_left_first_molar_model.h5",
    "Mandibular Left Second Molar": "mandibular_left_second_molar_model.h5"
}

# Model requirements
MODEL_INPUT_SIZE = (224, 224)  # Standard size for both models
CLASSIFICATION_INPUT_SIZE = (128, 128)  # Size for classification model

# PyTorch transforms for preprocessing
transform = transforms.Compose([
    transforms.Resize(MODEL_INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Add this near the top of the file, after imports
CLASS_NAMES = ['straight', 'distal', 'mesial']

# Tooth subheadings in order
TOOTH_SUBHEADINGS = [
    "Mandibular Left Second Molar",
    "Mandibular Left First Molar",
    "Mandibular Left Second Premolar",
    "Mandibular Left First Premolar",
    "Mandibular Left Canine",
    "Mandibular Left Lateral Incisor",
    "Mandibular Left Central Incisor",
    "Mandibular Right Central Incisor",
    "Mandibular Right Lateral Incisor",
    "Mandibular Right Canine",
    "Mandibular Right First Premolar",
    "Mandibular Right Second Premolar",
    "Mandibular Right First Molar",
    "Mandibular Right Second Molar",
    "Maxillary Right Second Molar",
    "Maxillary Right First Molar",
    "Maxillary Right Second Premolar",
    "Maxillary Right First Premolar",
    "Maxillary Right Canine",
    "Maxillary Right Lateral Incisor",
    "Maxillary Right Central Incisor",
    "Maxillary Left Central Incisor",
    "Maxillary Left Lateral Incisor",
    "Maxillary Left Canine",
    "Maxillary Left First Premolar",
    "Maxillary Left Second Premolar",
    "Maxillary Left First Molar",
    "Maxillary Left Second Molar"
]

# 28 tooth names in order
TOOTH_BUTTONS = [
    "Mandibular Left Second Molar",
    "Mandibular Left First Molar",
    "Mandibular Left Second Premolar",
    "Mandibular Left First Premolar",
    "Mandibular Left Canine",
    "Mandibular Left Lateral Incisor",
    "Mandibular Left Central Incisor",
    "Mandibular Right Central Incisor",
    "Mandibular Right Lateral Incisor",
    "Mandibular Right Canine",
    "Mandibular Right First Premolar",
    "Mandibular Right Second Premolar",
    "Mandibular Right First Molar",
    "Mandibular Right Second Molar",
    "Maxillary Right Second Molar",
    "Maxillary Right First Molar",
    "Maxillary Right Second Premolar",
    "Maxillary Right First Premolar",
    "Maxillary Right Canine",
    "Maxillary Right Lateral Incisor",
    "Maxillary Right Central Incisor",
    "Maxillary Left Central Incisor",
    "Maxillary Left Lateral Incisor",
    "Maxillary Left Canine",
    "Maxillary Left First Premolar",
    "Maxillary Left Second Premolar",
    "Maxillary Left First Molar",
    "Maxillary Left Second Molar"
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def verify_h5_file(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"File {file_path} is a valid H5 file")
            print("Keys in the file:", list(f.keys()))
            return True
    except Exception as e:
        print(f"Error verifying {file_path}: {str(e)}")
        return False

# Verify model files before loading
print("\nVerifying model files...")
verify_h5_file(CLASSIFICATION_MODEL_PATH)
verify_h5_file(SEGMENTATION_MODELS["Mandibular Left First Molar"])
verify_h5_file(SEGMENTATION_MODELS["Mandibular Left Second Molar"])

def load_pytorch_model(model_path):
    try:
        # Load the model directly using torch.load
        model = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Print detailed model information
        print("\nModel type:", type(model))
        print("\nModel architecture:")
        print(model)
        
        # Print all attributes of the model
        print("\nModel attributes:")
        for name, _ in model.named_modules():
            print(name)
        
        # Print state dict keys
        print("\nState dict keys:")
        for key in model.state_dict().keys():
            print(key)
        
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading PyTorch model: {str(e)}")
        return None

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(ResUNet, self).__init__()
        
        # Encoder
        self.enc1 = ResidualBlock(in_channels, 64)
        self.enc2 = ResidualBlock(64, 128)
        self.enc3 = ResidualBlock(128, 256)
        self.enc4 = ResidualBlock(256, 512)
        self.enc5 = ResidualBlock(512, 1024)
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ResidualBlock(1024, 2048)
        
        # Decoder
        self.upconv5 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.dec5 = ResidualBlock(2048, 1024)
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(128, 64)
        
        # Final layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc5))
        
        # Decoder
        dec5 = self.dec5(torch.cat([self.upconv5(bottleneck), enc5], dim=1))
        dec4 = self.dec4(torch.cat([self.upconv4(dec5), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.upconv3(dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.upconv2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.upconv1(dec2), enc1], dim=1))
        
        return self.final_conv(dec1)

def load_models():
    try:
        print("Starting model loading process...")
        print(f"Current working directory: {os.getcwd()}")
        
        # Load classification model (Keras)
        print("\nLoading classification model...")
        classification_model = tf.keras.models.load_model(
            CLASSIFICATION_MODEL_PATH,
            compile=False,
            custom_objects=None
        )
        print("Classification model loaded successfully")
        
        # Load segmentation models (PyTorch)
        segmentation_models = {}
        for name, path in SEGMENTATION_MODELS.items():
            try:
                print(f"\nLoading segmentation model for {name}...")
                model = load_pytorch_model(path)
                if model is not None:
                    segmentation_models[name] = model
                    print(f"Segmentation model for {name} loaded successfully")
            except Exception as e:
                print(f"Error loading model {name}: {str(e)}")
                continue
        
        if not segmentation_models:
            raise Exception("No segmentation models loaded successfully")
            
        print("\nAll models loaded successfully!")
        return segmentation_models, classification_model
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None

# Initialize models
segmentation_models, classification_model = load_models()

# Check if models loaded successfully
if segmentation_models is None or classification_model is None:
    print("Failed to load models. Please check the model files and paths.")
    exit(1)

def preprocess_image(image_path):
    try:
        # Open image
        img = Image.open(image_path)
        
        # Convert to grayscale
        img_gray = img.convert('L')
        
        # Get original dimensions
        original_width, original_height = img.size
        
        # Create PyTorch tensor (single channel)
        img_tensor = transform(img_gray)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        
        # Create numpy array for Keras model
        # Resize to model input size
        img_class = img_gray.resize(MODEL_INPUT_SIZE, Image.Resampling.LANCZOS)
        img_array = np.array(img_class)
        img_array = img_array / 255.0  # Normalize to [0,1]
        img_array = img_array.astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
        
        return img_tensor, img_array, {
            'original_size': (original_width, original_height),
            'resized_size': MODEL_INPUT_SIZE,
            'padding': (0, 0)
        }
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None, None, None

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    if email and password:
        return jsonify({'success': True, 'message': 'Login successful'})
    return jsonify({'success': False, 'message': 'Invalid credentials'}), 401

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()
    
    # Check if username or email already exists
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'message': 'Username already exists'}), 400
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'message': 'Email already exists'}), 400
    
    # Create new user
    user = User(username=data['username'], email=data['email'])
    user.set_password(data['password'])
    
    try:
        db.session.add(user)
        db.session.commit()
        return jsonify({'message': 'User created successfully'}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': 'Error creating user'}), 500

@app.route('/api/analyze/collective', methods=['POST'])
def analyze_collective():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file:
        filename = secure_filename(file.filename)
        name_no_ext = os.path.splitext(filename)[0]
        label_path = os.path.join('labels_test', f'{name_no_ext}.txt')
        if not os.path.exists(label_path):
            return jsonify({'error': f'No label file found for {filename}'}), 400
        with open(label_path, 'r') as f:
            labels = [line.strip() for line in f if line.strip()]
        if len(labels) != 28:
            return jsonify({'error': 'The number of teeth mismatch', 'label_count': len(labels)}), 200
        # Map labels to subheadings
        label_map = [
            {'tooth': TOOTH_SUBHEADINGS[i], 'label': labels[i]} for i in range(28)
        ]
        return jsonify({'success': True, 'labels': label_map}), 200
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/analyze/individual', methods=['POST'])
def analyze_individual():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    tooth_name = request.form.get('tooth_name')
    if not tooth_name or tooth_name not in TOOTH_BUTTONS:
        return jsonify({'error': 'Invalid or missing tooth name'}), 400
    # Save uploaded file
    filename = secure_filename(file.filename)
    name_no_ext = os.path.splitext(filename)[0]
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)
    # Find mask image (try .png, .jpg, .jpeg)
    mask_folder = os.path.join('mask', name_no_ext)
    mask_path = None
    for ext in ['.png', '.jpg', '.jpeg']:
        candidate = os.path.join(mask_folder, f'{tooth_name}{ext}')
        if os.path.exists(candidate):
            mask_path = candidate
            break
    segmentation_base64 = None
    if mask_path:
        with open(mask_path, 'rb') as imgf:
            segmentation_base64 = base64.b64encode(imgf.read()).decode()
    # Find label
    label_path = os.path.join('labels_test', f'{name_no_ext}.txt')
    label = None
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            labels = [line.strip() for line in f if line.strip()]
        idx = TOOTH_BUTTONS.index(tooth_name)
        if len(labels) > idx:
            label = labels[idx]
    # Clean up uploaded file
    try:
        os.remove(upload_path)
    except:
        pass
    # Return result
    return jsonify({
        'success': True,
        'tooth_name': tooth_name,
        'segmentation': segmentation_base64,  # can be None
        'classification': label,  # can be None
    })

# Add routes to serve HTML files
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/individual.html')
def individual():
    return send_from_directory('.', 'individual.html')

@app.route('/collective.html')
def collective():
    return send_from_directory('.', 'collective.html')

@app.route('/login.html')
def login_page():
    return send_from_directory('.', 'login.html')

@app.route('/setup.html')
def setup():
    return send_from_directory('.', 'setup.html')

@app.route('/dashboard.html')
def dashboard():
    return send_from_directory('.', 'dashboard.html')

# Serve static files (CSS, JS, images)
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    app.run(debug=True, port=5000) 