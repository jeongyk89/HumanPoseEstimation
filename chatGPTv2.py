import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import mpld3
import seaborn as sns

# Define the Dark2 color palette
dark2_palette = sns.color_palette("Dark2", 4)

# Create a new empty graph
G = nx.Graph()

# Add nodes to the graph
G.add_node('A')
G.add_node('B')
G.add_node('C')
G.add_node('D')

# Add edges to the graph
G.add_edge('A', 'B')
G.add_edge('B', 'C')
G.add_edge('C', 'D')
G.add_edge('D', 'A')

#write a code for calculation


# Define the function for handling mouse click events
def on_click(event):
    if event.inaxes is not None:
        node = None
        for n in G.nodes():
            if G.nodes[n]['xy'] is not None and G.nodes[n]['xy'].contains(event)[0]:
                node = n
                break
        if node is not None:
            pos = nx.get_node_attributes(G, 'pos')
            x, y = pos[node]
            dx = np.random.uniform(-0.1, 0.1)
            dy = np.random.uniform(-0.1, 0.1)
            G.nodes[node]['pos'] = (x + dx, y + dy)
            update_graph()

# Define the function for updating the graph
def update_graph():
    pos = nx.get_node_attributes(G, 'pos')
    color_map = []
    for n in G.nodes():
        if n == 'A':
            color_map.append(dark2_palette[0])
        elif n == 'B':
            color_map.append(dark2_palette[1])
        elif n == 'C':
            color_map.append(dark2_palette[2])
        else:
            color_map.append(dark2_palette[3])
    sc.set_offsets(np.array(list(pos.values())))
    sc.set_color(color_map)
    plt.draw()

# Draw the graph using Matplotlib
fig, ax = plt.subplots()
pos = nx.spring_layout(G, dim=2)
nx.set_node_attributes(G, pos, 'pos')
nx.set_node_attributes(G, None, 'xy')
sc = ax.scatter([pos[n][0] for n in G.nodes()], [pos[n][1] for n in G.nodes()], s=50, alpha=0.6, c=dark2_palette)
for n in G.nodes():
    xy = ax.annotate(n, (pos[n][0], pos[n][1]), color='white', fontsize=16, ha='center', va='center')
    G.nodes[n]['xy'] = xy
fig.canvas.mpl_connect('button_press_event', on_click)

# Convert the plot to an HTML page
html = mpld3.fig_to_html(fig)
with open('network.html', 'w') as f:
    f.write(html)
