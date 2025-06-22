import os
import random
import shutil

input_tagged = path...
input_ori = path...

output_base = path...
split_sizes = {'a': x, 'b': x}

output_dirs = {
    split: {
        'tagged_manual': os.path.join(output_base, split, 'tagged_manual'),
        'original_l3': os.path.join(output_base, split, 'original_l3')
    }
    for split in split_sizes
}

for dirs in output_dirs.values():
    os.makedirs(dirs['tagged_manual'], exist_ok=True)
    os.makedirs(dirs['original_l3'], exist_ok=True)

tagged_files = [f for f in os.listdir(input_tagged) if f.endswith('.tag')]
ori_files = os.listdir(input_ori)

ori_map = {}
for f in ori_files:
    base, ext = os.path.splitext(f)
    if base not in ori_map:  # first match wins
        ori_map[base] = f

valid_pairs = []
for tag_file in tagged_files:
    base = os.path.splitext(tag_file)[0]
    if base in ori_map:
        valid_pairs.append((base, tag_file, ori_map[base]))
    
print(f"{len(valid_pairs)} pairs gevonden")

total_required = sum(split_sizes.values())
if len(valid_pairs) < total_required:
    raise ValueError(f"{len(valid_pairs)} pairs gevonden, maar {total_required} nodig.")

random.seed(42)
random.shuffle(valid_pairs)

split_indices = {
    'a': valid_pairs[:split_sizes['b2val']],
    'b': valid_pairs[split_sizes['a']:split_sizes['a'] + split_sizes['b']],
}

for split, records in split_indices.items():
    for base, tag_file, ori_file in records:
        shutil.copy2(os.path.join(input_tagged, tag_file), os.path.join(output_dirs[split]['tagged_manual'], tag_file))
        shutil.copy2(os.path.join(input_ori, ori_file), os.path.join(output_dirs[split]['original_l3'], ori_file))
    print(f"Copied {len(records)} pairs to split '{split}'")

print("Dataset successfully split into b2val, b2train.")
