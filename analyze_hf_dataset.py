import argparse
from datasets import load_dataset
from tqdm import tqdm
from collections import Counter
import PIL

def analyze_dataset(dataset_name, split="train"):
    try:
        # Use imagefolder builder which handles zip files automatically
        dataset = load_dataset("imagefolder", data_files={"train": "*.zip"}, split=split, streaming=True)
    except Exception as e:
        print(f"Streaming failed, trying normal load: {e}")
        dataset = load_dataset(dataset_name, split=split)

    print("Analyzing images...")
    
    count = 0
    resolutions = Counter()
    modes = Counter()
    
    # We can't use len() on a streaming dataset easily without iterating
    # So we iterate through everything.
    
    for item in tqdm(dataset):
        image = item.get('image')
        if image:
            # Check if it's a PIL image or path (streaming usually gives PIL)
            if not isinstance(image, PIL.Image.Image):
                continue
                
            count += 1
            resolutions[image.size] += 1
            modes[image.mode] += 1
            
    print("\n" + "="*40)
    print(f"Dataset: {dataset_name}")
    print("="*40)
    print(f"Total Images: {count}")
    
    print("\nImage Modes:")
    for mode, c in modes.most_common():
        print(f"  {mode}: {c} ({c/count*100:.2f}%)")
        
    print("\nImage Resolutions (Width, Height):")
    # Print top 10 resolutions
    for res, c in resolutions.most_common(20):
        print(f"  {res}: {c} ({c/count*100:.2f}%)")
        
    if len(resolutions) > 20:
        print(f"  ... and {len(resolutions) - 20} other unique resolutions.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a Hugging Face dataset.")
    parser.add_argument("dataset", type=str, help="Name of the dataset (e.g., sm12377/trImgs)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to analyze")
    args = parser.parse_args()
    
    analyze_dataset(args.dataset, args.split)
