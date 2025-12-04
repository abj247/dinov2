import subprocess
import re

def get_node_info():
    try:
        output = subprocess.check_output(['scontrol', 'show', 'node'], universal_newlines=True)
        return output
    except subprocess.CalledProcessError as e:
        print(f"Error running scontrol: {e}")
        return ""

def parse_nodes(output):
    nodes = []
    # Split by double newline or NodeName start
    raw_nodes = output.split('\n\n')
    
    for raw_node in raw_nodes:
        if 'NodeName=' not in raw_node:
            continue
            
        node = {}
        # Extract NodeName
        name_match = re.search(r'NodeName=(\S+)', raw_node)
        if name_match:
            node['name'] = name_match.group(1)
        
        # Extract State
        state_match = re.search(r'State=(\S+)', raw_node)
        if state_match:
            node['state'] = state_match.group(1)
            
        # Extract Total GPUs from Gres
        gres_match = re.search(r'Gres=gpu:(\w+):(\d+)', raw_node)
        if gres_match:
            node['gpu_type'] = gres_match.group(1)
            node['total_gpus'] = int(gres_match.group(2))
        else:
            # Check for generic gpu:N
            gres_match_generic = re.search(r'Gres=gpu:(\d+)', raw_node)
            if gres_match_generic:
                node['gpu_type'] = 'generic'
                node['total_gpus'] = int(gres_match_generic.group(1))
            else:
                node['total_gpus'] = 0
                
        # Extract Allocated GPUs from AllocTRES
        alloc_match = re.search(r'AllocTRES=.*gres/gpu=(\d+)', raw_node)
        if alloc_match:
            node['alloc_gpus'] = int(alloc_match.group(1))
        else:
            node['alloc_gpus'] = 0
            
        nodes.append(node)
    return nodes

def main():
    output = get_node_info()
    nodes = parse_nodes(output)
    
    print(f"{'Node':<15} {'Type':<10} {'Total':<6} {'Alloc':<6} {'Free':<6} {'State':<10}")
    print("-" * 60)
    
    found = False
    for node in nodes:
        if node['total_gpus'] > 0:
            free_gpus = node['total_gpus'] - node['alloc_gpus']
            # Check if node is down or drained
            if 'DOWN' in node.get('state', '') or 'DRAIN' in node.get('state', '') or 'MAINT' in node.get('state', ''):
                continue
                
            if free_gpus >= 1:
                print(f"{node['name']:<15} {node.get('gpu_type', 'N/A'):<10} {node['total_gpus']:<6} {node['alloc_gpus']:<6} {free_gpus:<6} {node.get('state', 'N/A'):<10}")
                found = True
                
    if not found:
        print("No nodes found with >= 3 GPUs available.")

if __name__ == "__main__":
    main()
