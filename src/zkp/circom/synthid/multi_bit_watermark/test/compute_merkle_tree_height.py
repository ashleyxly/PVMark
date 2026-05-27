import math

def calculate_height(leaf_count):
    if leaf_count <= 0:
        return 0
    
    h1 = int(math.log2(leaf_count))
    lower_bound = 2 ** h1
    upper_bound = 2 ** (h1 + 1)
    
    # If the number of leaves is exactly a power of two, height is h1 + 1
    if leaf_count == upper_bound // 2:
        return h1 + 1
    else:
        return h1 + 1

def tree_height(leaves):
    if leaves < 1:
        return 0
    h = 1
    while (1 << (h - 1)) < leaves:
        h += 1
    return h



# Example usage:
leaf_count = 60000
height = tree_height(leaf_count)
print(f"The height of the complete binary tree with {leaf_count} leaves is {height}.")

