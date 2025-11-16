import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_image_dataloaders(base_dir, batch_size=16):
    """
    base_dir structure:
      train/real, train/fake, val/real, val/fake, test/real, test/fake
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    data = {
        phase: datasets.ImageFolder(root=os.path.join(base_dir, phase),
                                    transform=transform)
        for phase in ['train', 'val', 'test']
    }

    dataloaders = {
        phase: DataLoader(data[phase], batch_size=batch_size, shuffle=True)
        for phase in ['train', 'val', 'test']
    }

    return dataloaders, data['train'].classes
if __name__ == "__main__":
    base_dir = r"C:\Users\ashta\deepfake_detector\dataset\Celeb-DF Preprocessed"
    dataloaders, classes = get_image_dataloaders(base_dir)
    print(" Classes found:", classes)
    print(" Train batches:", len(dataloaders['train']))
    print(" Validation batches:", len(dataloaders['val']))
    print(" Test batches:", len(dataloaders['test']))
