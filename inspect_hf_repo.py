import argparse
from huggingface_hub import list_repo_files

def inspect_repo(repo_id):
    print(f"Inspecting repository: {repo_id}...")
    try:
        files = list_repo_files(repo_id, repo_type="dataset")
        print(f"Found {len(files)} files:")
        for f in files:
            print(f" - {f}")
    except Exception as e:
        print(f"Error inspecting repo: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", type=str)
    args = parser.parse_args()
    inspect_repo(args.repo_id)
