import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import statsmodels.api as sm

# ==========================================
# 1. DATA GENERATION STAGE
# ==========================================
def generate_mock_data(filename="sem_data.csv", sample_size=250):
    """
    Generates a synthetic dataset containing 5 Independent Variables (IV1 to IV5)
    and 1 Dependent Variable (DV) based on a known linear regression equation.
    """
    print(f"Generating synthetic dataset with {sample_size} samples...")
    np.random.seed(42)  # Set seed for reproducible results

    # Generate independent variables (IV1 to IV5) with standard normal distribution
    iv1 = np.random.normal(5, 1.5, sample_size)
    iv2 = np.random.normal(10, 2.0, sample_size)
    iv3 = np.random.normal(3, 0.8, sample_size)
    iv4 = np.random.normal(8, 1.2, sample_size)
    iv5 = np.random.normal(12, 2.5, sample_size)
    
    # Introduce random noise/residual error
    error = np.random.normal(0, 1.0, sample_size)

    # Calculate Dependent Variable using predefined true beta coefficients
    # Formula: DV = Intercept + b1*IV1 + b2*IV2 + b3*IV3 + b4*IV4 + b5*IV5 + Error
    dv = 2.5 + (0.45 * iv1) - (0.30 * iv2) + (0.65 * iv3) + (0.05 * iv4) + (0.22 * iv5) + error

    # Construct and export DataFrame
    df = pd.DataFrame({
        'IV1': iv1, 'IV2': iv2, 'IV3': iv3, 'IV4': iv4, 'IV5': iv5, 'DV': dv
    })
    
    df.to_csv(filename, index=False)
    print(f"Dataset successfully exported to: '{os.path.abspath(filename)}'\n")
    return filename


# ==========================================
# 2. STATISTICAL ANALYSIS STAGE
# ==========================================
def compute_path_analysis(csv_file):
    """
    Reads the CSV and fits an OLS regression model to calculate standardized 
    path coefficients (Beta weights) and statistical significance.
    """
    df = pd.read_csv(csv_file)
    
    # Standardize data to obtain standardized path coefficients (Beta weights)
    df_z = (df - df.mean()) / df.std()
    
    X = df_z[['IV1', 'IV2', 'IV3', 'IV4', 'IV5']]
    X = sm.add_constant(X)  # Add intercept
    y = df_z['DV']
    
    # Fit Ordinary Least Squares model
    model = sm.OLS(y, X).fit()
    
    results = {}
    for var in ['IV1', 'IV2', 'IV3', 'IV4', 'IV5']:
        results[var] = {
            'coefficient': model.params[var],
            'p_value': model.pvalues[var]
        }
    
    # Store R-squared value for display
    results['R2'] = model.rsquared
    return results


# ==========================================
# 3. GRAPHICAL VISUALIZATION STAGE
# ==========================================
def draw_sem_diagram(analysis_results):
    """
    Generates a structured Path Diagram mapping relationships from the 
    5 Independent Variables to the Dependent Variable using NetworkX.
    """
    print("Plotting SEM Structural Diagram...")
    G = nx.DiGraph()
    
    # Define variables
    iv_nodes = ['IV1', 'IV2', 'IV3', 'IV4', 'IV5']
    dv_node = 'DV'
    
    # Build a directional canvas layout
    pos = {}
    # Space out Independent Variables vertically on the left column (X = 0)
    for i, iv in enumerate(iv_nodes):
        pos[iv] = (0, 10 - (i * 2.5))
    # Place the Dependent Variable centered on the right column (X = 2)
    pos[dv_node] = (2, 5)
    
    plt.figure(figsize=(10, 7))
    
    # 1. Draw Latent/Observed Nodes
    nx.draw_networkx_nodes(G, pos, nodelist=iv_nodes, node_color='#D6EAF8', 
                           node_shape='o', node_size=2200, edgecolors='#2E86C1', linewidths=1.5)
    nx.draw_networkx_nodes(G, pos, nodelist=[dv_node], node_color='#D4EFDF', 
                           node_shape='o', node_size=2600, edgecolors='#27AE60', linewidths=2.0)
    
    # 2. Draw Text Labels for Variables
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold', font_family='sans-serif')
    
    # 3. Add Edges and Edge Labels containing Statistical Coefficients
    for iv in iv_nodes:
        G.add_edge(iv, dv_node)
        
        coef = analysis_results[iv]['coefficient']
        p_val = analysis_results[iv]['p_value']
        
        # Determine significance stars
        stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else " (ns)"
        edge_label = f"β = {coef:.2f}{stars}"
        
        # Draw explicit directional paths
        nx.draw_networkx_edges(G, pos, edgelist=[(iv, dv_node)], width=1.5, 
                               edge_color='#566573', arrowstyle='->', arrowsize=20)
        
        # Calculate coordinate positions for floating text labels along the paths
        x_mid = (pos[iv][0] + pos[dv_node][0]) / 2.0
        y_mid = (pos[iv][1] + pos[dv_node][1]) / 2.0
        
        # Slightly offset text markers horizontally to guarantee scannability
        plt.text(x_mid - 0.15, y_mid + 0.1, edge_label, fontsize=10, 
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="silver", lw=0.5, alpha=0.9))

    # Add R-Squared annotation block inside the plot area
    r2_val = analysis_results['R2']
    plt.text(1.8, 2.5, f"Model Variance\nR² = {r2_val:.3f}", fontsize=11, fontweight='bold',
             bbox=dict(boxstyle="square,pad=0.4", fc="#FCF3CF", ec="#F39C12", lw=1.5))

    # Final Canvas clean-up
    plt.title("Structural Equation Modeling (SEM) - Path Analysis Demonstration", fontsize=14, fontweight='bold', pad=20)
    plt.text(0.0, 0.2, "* p < 0.05, ** p < 0.01, *** p < 0.001, ns = non-significant", fontsize=9, style='italic', color='#7F8C8D')
    plt.axis('off')
    plt.tight_layout()
    
    # Save chart output and display frame
    output_img = "sem_path_diagram.png"
    plt.savefig(output_img, dpi=300)
    print(f"Path diagram graphic successfully saved to: '{os.path.abspath(output_img)}'")
    plt.show()


# ==========================================
# 4. ENTRY POINT PIPELINE
# ==========================================
if __name__ == "__main__":
    csv_filename = "sem_data.csv"
    
    # Execute Pipeline Steps
    generate_mock_data(csv_filename, sample_size=300)
    metrics = compute_path_analysis(csv_filename)
    draw_sem_diagram(metrics)