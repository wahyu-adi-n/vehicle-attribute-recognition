from torch import device, cuda
from torchvision import transforms as T

CAR_SPECS_DIR = '../data/Cars'

IMAGE_SIZE = 224

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

N_WAY = 5
N_SHOT = 5
N_QUERY = 10
N_WORKERS = 2
N_TASK = 200
N_TRAINING_EPISODES = 500
N_VALIDATION_TASK = 100

EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
DEVICE = device('cuda' if cuda.is_available() else 'cpu')
MODEL_DIR = '.weights/fsl.pth'


# COLOR
COLOR_MODEL_PATH = './weights/color_resnet.pt'

TRANSFORMS = T.Compose([
    T.Resize(size=(256, 256)),
    T.CenterCrop(size=(256, 256)),
    T.ToTensor()]
)

LABELS = ['black', 'blue', 'gray',  'red', 'white']
N_CLASS = len(LABELS)

SAVE_PLOT_PATH = "output/save_plot/"
SAVE_CM_PATH = "output/save_cm/"
