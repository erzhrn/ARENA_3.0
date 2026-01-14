"""
Interactive visualization for the apply_scale vector decomposition procedure.

This visualization shows how a residual stream vector is decomposed into:
- A parallel component (along flip_dir)
- An orthogonal component (perpendicular to flip_dir)

And how scaling affects only the parallel component.

Run this in a Jupyter notebook or use: python apply_scale_visualization.py
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_apply_scale_visualization():
    """
    Creates an interactive 2D visualization of the apply_scale procedure.
    
    In 2D, we can clearly see:
    - The original residual vector (resid)
    - The flip direction (v, normalized)
    - The parallel component (α × v)
    - The orthogonal component (β × w)
    - The result after applying scale: (-scale × α × v) + (β × w)
    """
    
    # Create figure with slider
    fig = go.Figure()
    
    # Define the vectors (in 2D for visualization)
    # flip_dir (v) - the direction we care about
    flip_dir = np.array([1.0, 0.5])
    flip_dir_normalized = flip_dir / np.linalg.norm(flip_dir)
    
    # Original residual vector
    resid = np.array([2.0, 1.5])
    
    # Compute the decomposition
    # α = projection of resid onto normalized flip_dir
    alpha = np.dot(resid, flip_dir_normalized)
    
    # Parallel component: α × v_normalized
    parallel_component = alpha * flip_dir_normalized
    
    # Orthogonal component: resid - parallel = β × w
    orthogonal_component = resid - parallel_component
    
    # Create traces for each scale value
    scale_values = np.linspace(-2, 3, 51)
    
    for i, scale in enumerate(scale_values):
        visible = (i == 25)  # Start with scale = 0.5 (middle-ish)
        
        # Compute the transformed result
        # result = (-scale × α × v_normalized) + orthogonal_component
        transformed_parallel = -scale * alpha * flip_dir_normalized
        result = transformed_parallel + orthogonal_component
        
        # Origin point
        origin = np.array([0, 0])
        
        # --- TRACES ---
        
        # 1. Flip direction (extended line for reference)
        t_line = np.linspace(-3, 3, 100)
        flip_line_x = t_line * flip_dir_normalized[0]
        flip_line_y = t_line * flip_dir_normalized[1]
        fig.add_trace(go.Scatter(
            x=flip_line_x, y=flip_line_y,
            mode='lines',
            line=dict(color='gray', width=1, dash='dash'),
            name='flip_dir line',
            visible=visible,
            showlegend=False
        ))
        
        # 2. Orthogonal direction line (for reference)
        ortho_dir = np.array([-flip_dir_normalized[1], flip_dir_normalized[0]])
        ortho_line_x = t_line * ortho_dir[0]
        ortho_line_y = t_line * ortho_dir[1]
        fig.add_trace(go.Scatter(
            x=ortho_line_x, y=ortho_line_y,
            mode='lines',
            line=dict(color='lightgray', width=1, dash='dot'),
            name='orthogonal line',
            visible=visible,
            showlegend=False
        ))
        
        # 3. Original residual vector (resid)
        fig.add_trace(go.Scatter(
            x=[origin[0], resid[0]], y=[origin[1], resid[1]],
            mode='lines+markers',
            line=dict(color='blue', width=3),
            marker=dict(size=[0, 10], symbol=['circle', 'arrow'], 
                       angleref='previous'),
            name='Original resid (α×v + β×w)',
            visible=visible,
            hovertemplate=f'Original resid<br>({resid[0]:.2f}, {resid[1]:.2f})<extra></extra>'
        ))
        
        # 4. Parallel component (α × v)
        fig.add_trace(go.Scatter(
            x=[origin[0], parallel_component[0]], 
            y=[origin[1], parallel_component[1]],
            mode='lines+markers',
            line=dict(color='red', width=2),
            marker=dict(size=[0, 8]),
            name=f'Parallel (α×v), α={alpha:.2f}',
            visible=visible,
            hovertemplate=f'Parallel component<br>α = {alpha:.2f}<extra></extra>'
        ))
        
        # 5. Orthogonal component (β × w) - shown from origin
        fig.add_trace(go.Scatter(
            x=[origin[0], orthogonal_component[0]], 
            y=[origin[1], orthogonal_component[1]],
            mode='lines+markers',
            line=dict(color='green', width=2),
            marker=dict(size=[0, 8]),
            name='Orthogonal (β×w)',
            visible=visible,
            hovertemplate=f'Orthogonal component<br>({orthogonal_component[0]:.2f}, {orthogonal_component[1]:.2f})<extra></extra>'
        ))
        
        # 6. Transformed parallel component (-scale × α × v)
        fig.add_trace(go.Scatter(
            x=[origin[0], transformed_parallel[0]], 
            y=[origin[1], transformed_parallel[1]],
            mode='lines+markers',
            line=dict(color='orange', width=2, dash='dash'),
            marker=dict(size=[0, 8]),
            name=f'Transformed parallel (-scale×α×v)',
            visible=visible,
            hovertemplate=f'Transformed parallel<br>-scale×α = {-scale*alpha:.2f}<extra></extra>'
        ))
        
        # 7. Result vector (transformed_parallel + orthogonal)
        fig.add_trace(go.Scatter(
            x=[origin[0], result[0]], y=[origin[1], result[1]],
            mode='lines+markers',
            line=dict(color='purple', width=3),
            marker=dict(size=[0, 12]),
            name='Result (-scale×α×v + β×w)',
            visible=visible,
            hovertemplate=f'Result<br>({result[0]:.2f}, {result[1]:.2f})<extra></extra>'
        ))
        
        # 8. Show construction: orthogonal + transformed_parallel = result
        # Draw from end of orthogonal to result
        fig.add_trace(go.Scatter(
            x=[orthogonal_component[0], result[0]], 
            y=[orthogonal_component[1], result[1]],
            mode='lines',
            line=dict(color='orange', width=1, dash='dot'),
            name='Construction line',
            visible=visible,
            showlegend=False
        ))
        
        # 9. Flip direction unit vector (v_normalized)
        fig.add_trace(go.Scatter(
            x=[origin[0], flip_dir_normalized[0]], 
            y=[origin[1], flip_dir_normalized[1]],
            mode='lines+markers',
            line=dict(color='black', width=2),
            marker=dict(size=[0, 10], symbol=['circle', 'triangle-up']),
            name='flip_dir (normalized)',
            visible=visible,
            hovertemplate='flip_dir (unit vector)<extra></extra>'
        ))
    
    # Number of traces per scale value
    traces_per_step = 9
    
    # Create slider steps
    steps = []
    for i, scale in enumerate(scale_values):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": f"apply_scale Visualization | scale = {scale:.2f}<br>" +
                           f"<sub>Transform: (α×v + β×w) → (-scale×α×v + β×w)</sub>"}],
            label=f"{scale:.1f}"
        )
        # Set visible for this step's traces
        for j in range(traces_per_step):
            step["args"][0]["visible"][i * traces_per_step + j] = True
        steps.append(step)
    
    # Add slider
    sliders = [dict(
        active=25,
        currentvalue={"prefix": "scale = ", "visible": True, "xanchor": "center"},
        pad={"t": 50},
        steps=steps
    )]
    
    # Update layout
    fig.update_layout(
        sliders=sliders,
        title=dict(
            text="apply_scale Visualization | scale = 0.5<br>" +
                 "<sub>Transform: (α×v + β×w) → (-scale×α×v + β×w)</sub>",
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            range=[-4, 4],
            title="x",
            scaleanchor="y",
            scaleratio=1,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='lightgray'
        ),
        yaxis=dict(
            range=[-3, 3],
            title="y",
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='lightgray'
        ),
        legend=dict(
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top'
        ),
        width=900,
        height=600,
        hovermode='closest'
    )
    
    # Add annotations explaining the math
    fig.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text="<b>The Math:</b><br>" +
             "1. Normalize: v̂ = v / ||v||<br>" +
             "2. Project: α = resid · v̂<br>" +
             "3. Parallel: α × v̂<br>" +
             "4. Orthogonal: resid - (α × v̂)<br>" +
             "5. Result: (-scale × α × v̂) + orthogonal",
        showarrow=False,
        font=dict(size=11),
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1,
        xanchor="left",
        yanchor="top"
    )
    
    return fig


def create_3d_visualization():
    """
    Creates a 3D version showing that the same principle applies in higher dimensions.
    """
    fig = go.Figure()
    
    # Define vectors in 3D
    flip_dir = np.array([1.0, 0.5, 0.3])
    flip_dir_normalized = flip_dir / np.linalg.norm(flip_dir)
    
    resid = np.array([2.0, 1.5, 1.0])
    
    # Decomposition
    alpha = np.dot(resid, flip_dir_normalized)
    parallel_component = alpha * flip_dir_normalized
    orthogonal_component = resid - parallel_component
    
    scale_values = np.linspace(-2, 3, 51)
    
    for i, scale in enumerate(scale_values):
        visible = (i == 25)
        
        transformed_parallel = -scale * alpha * flip_dir_normalized
        result = transformed_parallel + orthogonal_component
        
        origin = np.array([0, 0, 0])
        
        # Original resid
        fig.add_trace(go.Scatter3d(
            x=[origin[0], resid[0]], y=[origin[1], resid[1]], z=[origin[2], resid[2]],
            mode='lines+markers',
            line=dict(color='blue', width=5),
            marker=dict(size=[0, 6]),
            name='Original resid',
            visible=visible
        ))
        
        # Parallel component
        fig.add_trace(go.Scatter3d(
            x=[origin[0], parallel_component[0]], 
            y=[origin[1], parallel_component[1]], 
            z=[origin[2], parallel_component[2]],
            mode='lines+markers',
            line=dict(color='red', width=4),
            marker=dict(size=[0, 5]),
            name='Parallel (α×v)',
            visible=visible
        ))
        
        # Orthogonal component
        fig.add_trace(go.Scatter3d(
            x=[origin[0], orthogonal_component[0]], 
            y=[origin[1], orthogonal_component[1]], 
            z=[origin[2], orthogonal_component[2]],
            mode='lines+markers',
            line=dict(color='green', width=4),
            marker=dict(size=[0, 5]),
            name='Orthogonal (β×w)',
            visible=visible
        ))
        
        # Transformed parallel
        fig.add_trace(go.Scatter3d(
            x=[origin[0], transformed_parallel[0]], 
            y=[origin[1], transformed_parallel[1]], 
            z=[origin[2], transformed_parallel[2]],
            mode='lines+markers',
            line=dict(color='orange', width=4),
            marker=dict(size=[0, 5]),
            name='Transformed (-scale×α×v)',
            visible=visible
        ))
        
        # Result
        fig.add_trace(go.Scatter3d(
            x=[origin[0], result[0]], y=[origin[1], result[1]], z=[origin[2], result[2]],
            mode='lines+markers',
            line=dict(color='purple', width=5),
            marker=dict(size=[0, 6]),
            name='Result',
            visible=visible
        ))
        
        # flip_dir unit vector
        fig.add_trace(go.Scatter3d(
            x=[origin[0], flip_dir_normalized[0]], 
            y=[origin[1], flip_dir_normalized[1]], 
            z=[origin[2], flip_dir_normalized[2]],
            mode='lines+markers',
            line=dict(color='black', width=3),
            marker=dict(size=[0, 4]),
            name='flip_dir (unit)',
            visible=visible
        ))
    
    traces_per_step = 6
    
    steps = []
    for i, scale in enumerate(scale_values):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": f"3D Visualization | scale = {scale:.2f}"}],
            label=f"{scale:.1f}"
        )
        for j in range(traces_per_step):
            step["args"][0]["visible"][i * traces_per_step + j] = True
        steps.append(step)
    
    sliders = [dict(
        active=25,
        currentvalue={"prefix": "scale = ", "visible": True},
        pad={"t": 50},
        steps=steps
    )]
    
    fig.update_layout(
        sliders=sliders,
        title="3D apply_scale Visualization | scale = 0.5",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y", 
            zaxis_title="z",
            aspectmode='cube'
        ),
        width=800,
        height=700
    )
    
    return fig


def show_step_by_step():
    """
    Creates a step-by-step visualization showing each operation.
    """
    from plotly.subplots import make_subplots
    
    # Define vectors
    flip_dir = np.array([1.0, 0.5])
    flip_dir_normalized = flip_dir / np.linalg.norm(flip_dir)
    resid = np.array([2.0, 1.5])
    scale = 1.0
    
    alpha = np.dot(resid, flip_dir_normalized)
    parallel = alpha * flip_dir_normalized
    orthogonal = resid - parallel
    transformed_parallel = -scale * alpha * flip_dir_normalized
    result = transformed_parallel + orthogonal
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Step 1: Original resid",
            "Step 2: Decompose into parallel + orthogonal",
            "Step 3: Scale & flip the parallel component",
            "Step 4: Reconstruct result"
        ),
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )
    
    origin = [0, 0]
    
    # Common axis settings
    axis_range = [-3, 3]
    
    # Step 1: Show original resid and flip_dir
    fig.add_trace(go.Scatter(
        x=[origin[0], resid[0]], y=[origin[1], resid[1]],
        mode='lines+markers', line=dict(color='blue', width=3),
        marker=dict(size=[0, 10]), name='resid'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[origin[0], flip_dir_normalized[0]], y=[origin[1], flip_dir_normalized[1]],
        mode='lines+markers', line=dict(color='black', width=2),
        marker=dict(size=[0, 8]), name='flip_dir (normalized)'
    ), row=1, col=1)
    
    # Step 2: Decomposition
    fig.add_trace(go.Scatter(
        x=[origin[0], resid[0]], y=[origin[1], resid[1]],
        mode='lines+markers', line=dict(color='blue', width=2, dash='dot'),
        marker=dict(size=[0, 8]), name='resid', showlegend=False
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[origin[0], parallel[0]], y=[origin[1], parallel[1]],
        mode='lines+markers', line=dict(color='red', width=3),
        marker=dict(size=[0, 10]), name='parallel (α×v)'
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[parallel[0], resid[0]], y=[parallel[1], resid[1]],
        mode='lines+markers', line=dict(color='green', width=3),
        marker=dict(size=[0, 10]), name='orthogonal (β×w)'
    ), row=1, col=2)
    
    # Step 3: Transform parallel
    fig.add_trace(go.Scatter(
        x=[origin[0], parallel[0]], y=[origin[1], parallel[1]],
        mode='lines+markers', line=dict(color='red', width=2, dash='dot'),
        marker=dict(size=[0, 8]), name='original parallel', showlegend=False
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[origin[0], transformed_parallel[0]], y=[origin[1], transformed_parallel[1]],
        mode='lines+markers', line=dict(color='orange', width=3),
        marker=dict(size=[0, 10]), name='transformed (-scale×α×v)'
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[origin[0], orthogonal[0]], y=[origin[1], orthogonal[1]],
        mode='lines+markers', line=dict(color='green', width=2),
        marker=dict(size=[0, 8]), name='orthogonal (unchanged)', showlegend=False
    ), row=2, col=1)
    
    # Step 4: Reconstruct
    fig.add_trace(go.Scatter(
        x=[origin[0], resid[0]], y=[origin[1], resid[1]],
        mode='lines+markers', line=dict(color='blue', width=2, dash='dot'),
        marker=dict(size=[0, 8]), name='original', showlegend=False
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=[origin[0], orthogonal[0]], y=[origin[1], orthogonal[1]],
        mode='lines+markers', line=dict(color='green', width=2),
        marker=dict(size=[0, 8]), showlegend=False
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=[orthogonal[0], result[0]], y=[orthogonal[1], result[1]],
        mode='lines+markers', line=dict(color='orange', width=2),
        marker=dict(size=[0, 8]), showlegend=False
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=[origin[0], result[0]], y=[origin[1], result[1]],
        mode='lines+markers', line=dict(color='purple', width=3),
        marker=dict(size=[0, 12]), name='result'
    ), row=2, col=2)
    
    # Update all axes
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(range=axis_range, row=i, col=j, scaleanchor=f"y{(i-1)*2+j}")
            fig.update_yaxes(range=axis_range, row=i, col=j)
    
    fig.update_layout(
        title=dict(
            text="Step-by-Step: apply_scale Procedure (scale=1.0)",
            x=0.5, xanchor='center'
        ),
        width=900,
        height=800,
        showlegend=True,
        legend=dict(x=1.02, y=1)
    )
    
    return fig


# Main execution
if __name__ == "__main__":
    print("Creating interactive visualizations...")
    
    # Create and show the main 2D visualization
    fig_2d = create_apply_scale_visualization()
    fig_2d.write_html("apply_scale_2d_interactive.html")
    print("Saved: apply_scale_2d_interactive.html")
    
    # Create the step-by-step visualization
    fig_steps = show_step_by_step()
    fig_steps.write_html("apply_scale_steps.html")
    print("Saved: apply_scale_steps.html")
    
    # Create 3D visualization
    fig_3d = create_3d_visualization()
    fig_3d.write_html("apply_scale_3d_interactive.html")
    print("Saved: apply_scale_3d_interactive.html")
    
    print("\nOpen the HTML files in a browser, or run in Jupyter:")
    print("  from apply_scale_visualization import *")
    print("  create_apply_scale_visualization().show()")
    print("  show_step_by_step().show()")
    print("  create_3d_visualization().show()")
