import numpy as np


def augmentation_(x, insertion_rate=0.1, change_rate=0.1, p_break=0.2):
    n = len(x)
    #changes two 
    if np.random.rand(1) < p_break:
        x = x[len(x) // 2:] + x[:len(x) // 2]
    #elements deletion
    shift = np.random.randint(int(insertion_rate * n))
    indices_to_delete = np.random.choice(np.arange(len(x)-1), shift, replace=False)
    for i in sorted(indices_to_delete, reverse=True):
        if i >= len(x) - 1:
            print(sorted(indices_to_delete, reverse=True), i)
        else:
            del x[i]
    #elements insertion
    indices_to_insert = np.random.choice(np.arange(len(x)-1), shift)    
    new_elements = np.random.choice(["A", "C", "T", "G"], shift) 
    for i, idx in enumerate(sorted(indices_to_insert, reverse=False)):
        x.insert(idx + i, new_elements[i])
    #changing elements
    x = np.array(x)
    n_to_augment = np.random.randint(int(np.round(change_rate * n)))
    positions = np.random.randint(low=0, high=n, size=n_to_augment)
    random_chrom = np.random.choice(["A", "C", "T", "G"], n_to_augment)
    x[positions] = random_chrom
    x = list(x)
    return x


def augmentation_func(insertion_rate=0.1, change_rate=0.1, p_break=0.2):
    return lambda x: augmentation_(x, insertion_rate, change_rate, p_break)