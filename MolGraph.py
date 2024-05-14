from rdkit import Chem
import networkx as nx
import matplotlib.pyplot as plt

def MolGraph(smiles):
    # Convert SMILES to a molecule object
    mol = Chem.MolFromSmiles(smiles)

    if not mol:
        raise ValueError("Invalid SMILES string")

    # Create an empty graph
    G = nx.Graph()

    # Define colors for different atoms
    atom_color_map = {
        'C': 'skyblue',
        'O': 'red',
        'N': 'green',
        'H': 'gray',
        'S': 'yellow',
        'F': 'orange',
        'Cl': 'lime',
        'Br': 'maroon',
        'I': 'purple'
        # Add more atom types and colors as needed
    }

    # Add nodes with atom labels and colors from the molecule
    node_colors = []
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), label=atom.GetSymbol())
        node_colors.append(atom_color_map.get(atom.GetSymbol(), 'black'))  # Use black as default color

    # Add edges based on bonds
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    # Drawing options for the networkx graph
    pos = nx.spring_layout(G)  # positions for all nodes
    labels = nx.get_node_attributes(G, 'label')

    # Draw the graph
    nx.draw(G, pos, labels=labels, with_labels=True, node_color=node_colors, node_size=700, edge_color='k', linewidths=1, font_size=15)
    plt.show()
