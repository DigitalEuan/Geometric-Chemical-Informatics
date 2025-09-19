#!/usr/bin/env python3
"""
UBP Interactive Periodic Neighborhood Explorer
Phase 7: Create interactive visualization of the geometric mapping results

This system generates an interactive HTML file that allows exploration of the
UBP-enhanced geometric map of inorganic materials.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json

class UBPExplorerGenerator:
    """Generates the interactive UBP Periodic Neighborhood map"""

    def __init__(self):
        self.title = "UBP Interactive Periodic Neighborhood Map"
        self.description = "An interactive UMAP projection of 495 inorganic materials encoded with the Universal Binary Principle framework."

    def create_explorer(self, ubp_data_file: str, geometric_results_file: str, output_file="ubp_periodic_neighborhood_map.html"):
        """Create and save the interactive HTML explorer"""

        print("="*80)
        print("UBP INTERACTIVE PERIODIC NEIGHBORHOOD EXPLORER")
        print("="*80)
        print("Phase 7: Creating Interactive Explorer and Final Report")
        print()

        # Load data
        print("Loading UBP-encoded data and geometric analysis results...")
        try:
            df = pd.read_csv(ubp_data_file)
            print(f"✅ Loaded {len(df)} UBP-encoded materials")

            with open(geometric_results_file, 'r') as f:
                geometric_results = json.load(f)
            print(f"✅ Loaded geometric analysis results")

            # Load UMAP embeddings from .npy file
            embeddings = np.load("ubp_umap_embeddings.npy")
            print(f"✅ Loaded UMAP embeddings for {len(embeddings)} materials from .npy file")

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None

        # Check for data consistency
        if len(df) != len(embeddings):
            print(f"❌ Data length mismatch: {len(df)} materials vs {len(embeddings)} embeddings")
            # Align data by taking the minimum length
            min_len = min(len(df), len(embeddings))
            df = df.iloc[:min_len]
            embeddings = embeddings[:min_len]
            print(f"  ⚠️  Data truncated to {min_len} samples")

        # Prepare data for plotting
        print("\nPreparing data for visualization...")
        plot_data = self._prepare_plot_data(df, embeddings)

        # Create interactive plot
        print("Creating interactive Plotly visualization...")
        fig = self._create_interactive_plot(plot_data)

        # Save to HTML
        print(f"Saving interactive map to {output_file}...")
        try:
            fig.write_html(output_file, full_html=True, include_plotlyjs='cdn')
            print(f"✅ Successfully saved interactive map to {output_file}")
        except Exception as e:
            print(f"❌ Error saving HTML file: {e}")

        print("\n✅ Interactive explorer generation complete!")
        return output_file

    def _prepare_plot_data(self, df, embeddings):
        """Prepare data for Plotly visualization"""

        # Create hover text
        hover_text = []
        for i, row in df.iterrows():
            text = f"<b>{row.get('formula', 'N/A')}</b> (ID: {row.get('material_id', 'N/A')})<br>" \
                   f"--------------------------------------------------<br>" \
                   f"<b>Primary Realm:</b> {row.get('primary_realm', 'N/A')}<br>" \
                   f"<b>UBP Quality Score:</b> {row.get('ubp_quality_score', 0):.4f}<br>" \
                   f"<b>NRCI:</b> {row.get('nrci_calculated', 0):.6f}<br>" \
                   f"<b>System Coherence:</b> {row.get('system_coherence', 0):.4f}<br>" \
                   f"<b>Resonance Potential:</b> {row.get('total_resonance_potential', 0):.4f}<br>" \
                   f"<b>Dominant TM Element:</b> {row.get('tm_element', 'N/A')}<br>" \
                   f"<b>Resonance Cluster:</b> {row.get('resonance_cluster', 'N/A')}"
            hover_text.append(text)

        # Prepare color data
        color_data = {
            'primary_realm': df['primary_realm'].astype('category').cat.codes,
            'ubp_quality_score': df['ubp_quality_score'].values,
            'nrci_calculated': df['nrci_calculated'].values,
            'tm_element': df['tm_element'].astype('category').cat.codes,
            'resonance_cluster': df['resonance_cluster'].astype('category').cat.codes
        }

        plot_data = {
            'x': embeddings[:, 0],
            'y': embeddings[:, 1],
            'hover_text': hover_text,
            'color_data': color_data,
            'df': df
        }

        print("  ✅ Prepared hover text and color data")
        return plot_data

    def _create_interactive_plot(self, plot_data):
        """Create the interactive Plotly figure"""

        # Create main scatter plot trace
        trace = go.Scattergl(
            x=plot_data['x'],
            y=plot_data['y'],
            mode='markers',
            marker=dict(
                size=8,
                color=plot_data['color_data']['primary_realm'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title='Primary Realm',
                    tickvals=plot_data['df']['primary_realm'].astype('category').cat.codes.unique(),
                    ticktext=plot_data['df']['primary_realm'].astype('category').cat.categories
                ),
                opacity=0.8
            ),
            text=plot_data['hover_text'],
            hoverinfo='text',
            name='Materials'
        )

        # Create layout with dropdown menu for coloring
        layout = go.Layout(
            title=self.title,
            xaxis=dict(title='UMAP Dimension 1', showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(title='UMAP Dimension 2', showgrid=False, zeroline=False, showticklabels=False),
            hovermode='closest',
            template='plotly_dark',
            annotations=[
                dict(text=self.description, showarrow=False, xref='paper', yref='paper', x=0.005, y=1.08)
            ],
            updatemenus=[
                dict(
                    buttons=[
                        dict(
                            args=[{'marker.color': [plot_data['color_data']['primary_realm']],
                                   'marker.colorbar.title': 'Primary Realm',
                                   'marker.colorbar.tickvals': plot_data['df']['primary_realm'].astype('category').cat.codes.unique(),
                                   'marker.colorbar.ticktext': plot_data['df']['primary_realm'].astype('category').cat.categories}],
                            label="Color by Primary Realm",
                            method="restyle"
                        ),
                        dict(
                            args=[{'marker.color': [plot_data['color_data']['ubp_quality_score']],
                                   'marker.colorbar.title': 'UBP Quality Score'}],
                            label="Color by UBP Quality Score",
                            method="restyle"
                        ),
                        dict(
                            args=[{'marker.color': [plot_data['color_data']['nrci_calculated']],
                                   'marker.colorbar.title': 'NRCI'}],
                            label="Color by NRCI",
                            method="restyle"
                        ),
                        dict(
                            args=[{'marker.color': [plot_data['color_data']['tm_element']],
                                   'marker.colorbar.title': 'TM Element',
                                   'marker.colorbar.tickvals': plot_data['df']['tm_element'].astype('category').cat.codes.unique(),
                                   'marker.colorbar.ticktext': plot_data['df']['tm_element'].astype('category').cat.categories}],
                            label="Color by TM Element",
                            method="restyle"
                        ),
                        dict(
                            args=[{'marker.color': [plot_data['color_data']['resonance_cluster']],
                                   'marker.colorbar.title': 'Resonance Cluster',
                                   'marker.colorbar.tickvals': plot_data['df']['resonance_cluster'].astype('category').cat.codes.unique(),
                                   'marker.colorbar.ticktext': plot_data['df']['resonance_cluster'].astype('category').cat.categories}],
                            label="Color by Resonance Cluster",
                            method="restyle"
                        )
                    ],
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                )
            ]
        )

        fig = go.Figure(data=[trace], layout=layout)

        print("  ✅ Created interactive Plotly figure with dropdown menu")
        return fig

def main():
    """Main execution function"""

    print("Starting UBP Interactive Explorer Generation...")
    print("Phase 7: Creating Interactive Explorer and Final Report")
    print()

    # Initialize explorer generator
    generator = UBPExplorerGenerator()

    # Create the explorer
    output_file = generator.create_explorer(
        "ubp_encoded_inorganic_materials.csv",
        "ubp_geometric_analysis_results.json"
    )

    if output_file:
        print("\n" + "="*80)
        print("INTERACTIVE EXPLORER COMPLETE")
        print("="*80)
        print(f"✅ Successfully generated interactive map: {output_file}")
        print("✅ Ready to compile final scientific study report")
    else:
        print("❌ Interactive explorer generation failed")

if __name__ == "__main__":
    main()

