import matplotlib.pyplot as plt
import numpy as np

# This is improtant to see individual columns in a mask
# plt.rcParams['figure.dpi']= 600

def plot_prefill_masks(mask1, mask2, title):
    """
    Plot two prefill attention masks side by side with custom styling.
    
    Args:
        mask1: 2D boolean numpy array or tensor for left plot
        mask2: 2D boolean numpy array or tensor for right plot
        title1: Optional string to display as title for left plot
        title2: Optional string to display as title for right plot
    """
    # Create figure with gray background
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#3d3d3d')
    
    # Custom colormap (purple to yellow)
    custom_cmap = plt.cm.colors.ListedColormap(['#2E1A47', '#FFE135'])
    
    # Plot left mask
    ax1.imshow(~mask1, cmap=custom_cmap, aspect='equal')
    ax1.axis('off')
    ax1.set_title(f'{title} Mask #1', pad=5, color='white')
    ax1.text(-0.05, 0.5, 'Queries', rotation=90,
             transform=ax1.transAxes, va='center', color='white')
    ax1.text(0.5, -0.05, 'Keys', transform=ax1.transAxes, ha='center', color='white')
    ax1.set_facecolor('#3d3d3d')
    
    # Plot right mask
    ax2.imshow(~mask2, cmap=custom_cmap, aspect='equal')
    ax2.axis('off') 
    ax2.set_title(f'{title} Mask #2', pad=5, color='white')
    ax2.text(-0.05, 0.5, 'Queries', rotation=90,
             transform=ax2.transAxes, va='center', color='white')
    ax2.text(0.5, -0.05, 'Keys', transform=ax2.transAxes, ha='center', color='white')
    ax2.set_facecolor('#3d3d3d')

    plt.tight_layout()


def plot_generation_masks(mask1, mask2, title, mult):
    """
    Plot two generation attention masks stacked vertically with custom styling.
    
    Args:
        mask1: 2D boolean numpy array or tensor for top plot (N queries × M keys)
        mask2: 2D boolean numpy array or tensor for bottom plot (N queries × M keys)
        title: String to display as base title
        mult: Multiplier for height of the plot
    """
    # Create figure with gray background
    n_queries = mask1.shape[0]
    height = min(n_queries * mult, 10)  # Scale height with queries, bounded between 4 and 10
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, height))
    fig.patch.set_facecolor('#3d3d3d')
    
    # Custom colormap (purple for background False, yellow for True, red for generation False)
    colors = ['#2E1A47', '#FFE135', '#E74C3C']  # purple, yellow, red
    custom_cmap = plt.cm.colors.ListedColormap(colors)
    
    for mask, ax, mask_num in [(mask1, ax1, 1), (mask2, ax2, 2)]:
        # Create visualization array (0: background False, 1: True, 2: generation False)
        n_queries, n_keys = mask.shape
        n_gen_keys = n_queries  # Last N keys are generation keys
        
        viz_array = (~mask).astype(int)  # Convert to int (False -> 1, True -> 0)
        
        # Mark generation area with different color (2) if False
        gen_area = viz_array[:, -n_gen_keys:]
        viz_array[:, -n_gen_keys:] = np.where(gen_area == 1, 2, 0)
        
        # Plot mask
        ax.imshow(viz_array, cmap=custom_cmap, aspect='auto')
        ax.axis('off')
        ax.set_title(f'{title} Mask #{mask_num}', pad=5, color='white')
        ax.text(-0.02, 0.5, 'Queries', rotation=90,
                transform=ax.transAxes, va='center', color='white')
        ax.text(0.5, -0.3, 'Keys', transform=ax.transAxes, ha='center', color='white')
        ax.set_facecolor('#3d3d3d')
    
    plt.tight_layout()


def plot_sparse_attention_results(results):
    # Separate Dense results and get baseline
    dense_results = [r for r in results if 'Dense' in r['name']]
    sparse_results = [r for r in results if 'Dense' not in r['name']]
    
    # Calculate dense baseline
    dense_baseline = np.mean([r['loss'] for r in dense_results])
    
    # Group sparse results by method
    method_groups = {}
    for result in sparse_results:
        # Extract method name (remove 'Attention' suffix)
        method = result['name'].replace('Attention', '')
        
        if method not in method_groups:
            method_groups[method] = {'sparsity': [], 'loss': []}
            
        method_groups[method]['sparsity'].append(result['sparsity'])
        method_groups[method]['loss'].append(result['loss'])
    
    # Plot settings
    plt.figure(figsize=(10, 6))
    color_palette = ['#2ecc71', '#e74c3c', '#9b59b6', '#3498db', '#f1c40f']
    marker_palette = ['o', 's', 'D', '^', 'v']
    
    # Plot each method
    for i, (method, data) in enumerate(method_groups.items()):
        color_idx = i % len(color_palette)
        marker_idx = i % len(marker_palette)
        
        plt.plot(data['sparsity'], data['loss'],
                label=method,
                color=color_palette[color_idx],
                marker=marker_palette[marker_idx],
                linewidth=2,
                markersize=8)
    
    # Add dense baseline
    plt.axhline(y=dense_baseline, color='#95a5a6', linestyle='--', label='Dense')
    
    # Customize plot
    plt.xlabel('Sparsity Ratio', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Comparison of Sparse Attention Methods', fontsize=14, pad=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Set axis limits
    plt.xlim(-0.05, 1.05)
    losses = [l for data in method_groups.values() for l in data['loss']] + [dense_baseline]
    sparsity = [s for data in method_groups.values() for s in data['sparsity']]

    y_min, y_max = min(losses), max(losses)
    y_padding = (y_max - y_min) * 0.1
    plt.ylim(y_min - y_padding, y_max + y_padding)

    x_min, x_max = min(sparsity), max(sparsity)
    x_padding = (x_max - x_min) * 0.1
    plt.xlim(x_min - x_padding, x_max + x_padding)
    
    # Save plot
    plt.tight_layout()
