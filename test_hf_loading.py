import sys
import os
import torch
from torchvision import transforms

# Add current directory to path so we can import dinov2
sys.path.append(os.getcwd())

from dinov2.data.datasets.huggingface import HuggingFaceDataset
from dinov2.data.loaders import make_dataset

def test_hf_loading():
    print("Testing HuggingFaceDataset loading...")
    
    # Test direct instantiation
    try:
        dataset = HuggingFaceDataset(root='sm12377/tr_imgs', split='train')
        print(f"Successfully instantiated dataset. Size: {len(dataset)}")
        
        if len(dataset) > 0:
            image, target = dataset[0]
            print(f"Sample 0: Image size {image.size}, Target {target}")
            
            # Verify image is RGB
            if image.mode != 'RGB':
                print(f"WARNING: Image mode is {image.mode}, expected RGB")
            else:
                print("Image mode is RGB")
                
    except Exception as e:
        print(f"Failed to instantiate dataset directly: {e}")
        return

    # Test via factory
    print("\nTesting make_dataset factory...")
    try:
        dataset_str = "HuggingFace:root=sm12377/tr_imgs:split=train"
        dataset = make_dataset(dataset_str=dataset_str, transform=None)
        print(f"Successfully created dataset via factory: {dataset_str}")
        print(f"Dataset size: {len(dataset)}")
    except Exception as e:
        print(f"Failed to create dataset via factory: {e}")

if __name__ == "__main__":
    test_hf_loading()
