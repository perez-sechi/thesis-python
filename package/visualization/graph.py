import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def draw(
    psi, r, positive_alpha=0.0, negative_alpha=0.0, positive_beta=0.0,
    negative_beta=0.0, symmetric=True, arched=False, arch_radius=0.2,
    node_size_upper_limit=1500, layout='spring', label_margin=None,
    node_pos=None, label_pos=None, label_color='white',
    label_weight='normal', positive_color='black',
    negative_color='red', output_path=None, plot_margin=None,
    node_label_size_limit=500
):
    '''
    Draws the shapley matrices of Psi and R

    Parameters:
    psi (list(float)): Psi values
    r (list(list(float))): R values
    positive_alpha (float): alpha^{+} threshold
    negative_alpha (float): alpha^{-} threshold
    positive_beta (float): beta^{+} threshold
    negative_beta (float): beta^{-} threshold
    symmetric (bool): Defines the symmetry condition
    layout (str): Defines the Networkx drawing layout

    Returns:
    Graph: Networkx graph
    '''
    def filter_edge(x):
        if x > 0 and abs(x) > positive_beta:
            return x
        elif x < 0 and abs(x) > negative_beta:
            return x
        else:
            return 0

    def filter_node(x):
        if positive_alpha == 0 and negative_alpha == 0:
            return True
        elif x > 0 and abs(x) > positive_alpha:
            return True
        elif x < 0 and abs(x) > negative_alpha:
            return True
        else:
            return False

    r = r.copy().applymap(filter_edge)
    adjacency = r.copy().applymap(lambda x: 1 if x != 0 else 0)
    g = nx.from_pandas_adjacency(adjacency, create_using=nx.DiGraph)

    nodes = g.nodes()
    filtered_nodes = [
        n
        for n in nodes
        if not filter_node(psi['value'][n])
    ]
    g.remove_nodes_from(filtered_nodes)

    edges = g.edges()
    edge_color = [
        negative_color if r[v][u] < 0 else positive_color
        for u, v in edges
    ]

    edge_weight_upper_limit = 3
    edge_weights = [
        edge_weight_upper_limit * abs(r[v][u])
        for u, v in edges
    ]

    node_size_lower_limit = 0
    nodes = g.nodes()
    node_size_dict = {
        n: node_size_lower_limit +
        (node_size_upper_limit * abs(psi['value'][n]))
        for n in nodes
    }
    node_size = list(node_size_dict.values())

    node_color = [
        negative_color if psi['value'][n] < 0 else positive_color
        for n in nodes
    ]

    if node_pos == None:
        if layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(g)
        elif layout == 'circular':
            pos = nx.circular_layout(g)
        elif layout == 'spectral':
            pos = nx.spectral_layout(g)
        elif layout == 'spring':
            pos = nx.spring_layout(g)
        elif layout == 'planar':
            pos = nx.planar_layout(g)
        elif layout == 'shell':
            pos = nx.shell_layout(g)
        elif layout == 'spiral':
            pos = nx.spiral_layout(g)
        elif layout == 'twopi':
            pos = nx.nx_agraph.graphviz_layout(g, prog='twopi')
        elif layout == 'fruchterman_reingold_layout':
            pos = nx.fruchterman_reingold_layout(g)
        else:
            pos = nx.random_layout(g)
    else:
        pos = node_pos

    plt.figure()
    if plot_margin != None:
        plt.margins(x=plot_margin, y=plot_margin)
    nx.draw_networkx_nodes(
        g, pos,
        node_size=node_size,
        node_color=node_color
    )
    if label_pos == None:
        nx.draw_networkx_labels(
            g, pos,
            labels={node: node for node in g.nodes(
            ) if node_size_dict[node] > node_label_size_limit},
            font_size=10,
            font_color='white',
            font_weight='bold'
        )
        nx.draw_networkx_labels(
            g, pos,
            labels={node: node for node in g.nodes(
            ) if node_size_dict[node] == 0},
            font_size=10,
            font_color='black',
            font_weight='bold'
        )

        x_max = max([x for x, y in pos.values()])
        y_max = max([y for x, y in pos.values()])
        x_min = min([x for x, y in pos.values()])
        y_min = min([y for x, y in pos.values()])

        y_diff = y_max - y_min
        x_diff = x_max - x_min
        hipo = (y_diff ** 2 + x_diff ** 2) ** 0.5

        max_node_size = max(node_size)
        max_node_size_canvas_ratio = 8
        font_canvas_ratio = 500
        font_margin = 12 * hipo / 2 / font_canvas_ratio


        for node in [
            node for node in nodes
            if node_size_dict[node] <= node_label_size_limit
            and node_size_dict[node] > 0
        ]:
            x, y = pos[node]
            curr_node_size = node_size_dict[node]

            default_x_margin = curr_node_size * (x_diff / max_node_size / max_node_size_canvas_ratio)
            default_y_margin = curr_node_size * (y_diff / max_node_size  / max_node_size_canvas_ratio)

            x_margin = default_x_margin \
                if label_margin == None \
                else label_margin

            y_margin = default_y_margin \
                if label_margin == None \
                else label_margin

            x_margin_label = font_margin * (4 + len(str(node))) / 2
            y_margin_label = font_margin
            
            if y < 0:
                y_label_pos = y - y_margin - y_margin_label
            elif y > 0:
                y_label_pos = y + y_margin
            else:
                y_label_pos = y

            if x < 0:
                x_label_pos = x - x_margin - x_margin_label
            elif x > 0:
                x_label_pos = x + x_margin + x_margin_label
            else:
                x_label_pos = x

            plt.text(
                x_label_pos, y_label_pos, s=node,
                horizontalalignment='center'
            )
    else:
        nx.draw_networkx_labels(
            g, label_pos,
            labels={node: node for node in g.nodes()},
            font_size=10,
            font_color=label_color,
            font_weight=label_weight
        )

    if symmetric:
        if arched:
            nx.draw_networkx_edges(
                g, pos,
                edge_color=edge_color,
                width=edge_weights,
                node_size=[x + 400 for x in node_size],
                arrows=True,
                arrowsize=0,
                arrowstyle='|-|',
                connectionstyle=f'arc3,rad={arch_radius}',
            )
        else:
            nx.draw_networkx_edges(
                g, pos,
                edge_color=edge_color,
                width=edge_weights,
                node_size=[x + 400 for x in node_size],
                arrows=True,
                arrowsize=0,
                arrowstyle='|-|'
            )
    else:
        nx.draw_networkx_edges(
            g, pos,
            edge_color=edge_color,
            width=edge_weights,
            node_size=[x + 400 for x in node_size],
            connectionstyle=f'arc3,rad={arch_radius}',
        )

    plt.axis('off')
    plt.show()

    if output_path != None:
        print('Saving the graph to', output_path)
        plt.savefig(output_path, format='jpg', bbox_inches='tight')

    return g


def get_cmap(n, name='hsv'):
    '''
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.
    '''
    return plt.cm.get_cmap(name, n + 1)

def draw_clusters(
    psi, r, clusters=[], positive_alpha=0, negative_alpha=0, positive_beta=0,
    negative_beta=0, symmetric=True, arched=False, arch_radius=0.2,
    node_size_upper_limit=1500, layout='spring', label_margin=None,
    node_pos=None, label_pos=None, label_color='white',
    label_weight='normal', positive_color='black',
    negative_color='red', output_path=None, plot_margin=None,
    node_label_size_limit=500
):
    '''
    Draws the shapley matrices of Psi and R

    Parameters:
    psi (list(float)): Psi values
    r (list(list(float))): R values
    clusters (list(list(int))): List of clusters
    positive_alpha (float): alpha^{+} threshold
    negative_alpha (float): alpha^{-} threshold
    positive_beta (float): beta^{+} threshold
    negative_beta (float): beta^{-} threshold
    symmetric (bool): Defines the symmetry condition
    layout (str): Defines the Networkx drawing layout

    Returns:
    Graph: Networkx graph
    '''
    def filter_edge(x):
        if x > 0 and abs(x) > positive_beta:
            return x
        elif x < 0 and abs(x) > negative_beta:
            return x
        else:
            return 0

    def filter_node(x):
        if positive_alpha == 0 and negative_alpha == 0:
            return True
        elif x > 0 and abs(x) > positive_alpha:
            return True
        elif x < 0 and abs(x) > negative_alpha:
            return True
        else:
            return False

    r = r.copy().applymap(filter_edge)
    adjacency = r.copy().applymap(lambda x: 1 if x != 0 else 0)
    g = nx.from_pandas_adjacency(adjacency, create_using=nx.DiGraph)

    nodes = g.nodes()
    filtered_nodes = [
        n
        for n in nodes
        if not filter_node(psi['value'][n])
    ]
    g.remove_nodes_from(filtered_nodes)
    nodes = g.nodes()

    clusters = [
        [node for node in c if node in nodes]
        for c in clusters
    ]
    clusters = [
        c for c in clusters if len(c) > 0
    ]

    edges = g.edges()
    edge_color = [
        negative_color if r[v][u] < 0 else positive_color
        for u, v in edges
    ]

    edge_weight_upper_limit = 3
    edge_weights = [
        edge_weight_upper_limit * abs(r[v][u])
        for u, v in edges
    ]

    node_size_lower_limit = 0
    node_size_dict = {
        n: node_size_lower_limit +
        (node_size_upper_limit * abs(psi['value'][n]))
        for n in nodes
    }
    node_size = list(node_size_dict.values())

    node_color = [
        negative_color if psi['value'][n] < 0 else positive_color
        for n in nodes
    ]

    if node_pos == None:
        if layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(g)
        elif layout == 'circular':
            pos = nx.circular_layout(g)
        elif layout == 'spectral':
            pos = nx.spectral_layout(g)
        elif layout == 'spring':
            pos = nx.spring_layout(g)
        elif layout == 'planar':
            pos = nx.planar_layout(g)
        elif layout == 'shell':
            pos = nx.shell_layout(g)
        elif layout == 'spiral':
            pos = nx.spiral_layout(g)
        elif layout == 'twopi':
            pos = nx.nx_agraph.graphviz_layout(g, prog='twopi')
        elif layout == 'fruchterman_reingold_layout':
            pos = nx.fruchterman_reingold_layout(g)
        else:
            pos = nx.random_layout(g)
    else:
        pos = node_pos

    plt.figure()

    if plot_margin != None:
        plt.margins(x=plot_margin, y=plot_margin)

    if len(clusters) > 0:
        # superpos = copy.deepcopy(pos)
        superpos = nx.spring_layout(g, scale=2.5)

        centers = list(superpos.values())
        pos = {}
        for center, comm in zip(centers, clusters):
            pos.update(
                nx.spring_layout(
                    nx.subgraph(g, comm),
                    center=center
                )
            )

        legend_elements = []
        cmap = get_cmap(len(clusters))
        for cdx, cluster_nodes in enumerate(clusters):
            color = cmap(cdx)
            cluster_node_size = [node_size_dict[node] for node in cluster_nodes]
            nx.draw_networkx_nodes(
                g, pos=pos, nodelist=cluster_nodes,
                node_color=color, node_size=cluster_node_size
            )

            legend_elements.append(
                Line2D(
                    [0], [0], marker='o', color='white', label=f"Community {cdx + 1}",
                    markerfacecolor=color, markersize=10
                )
            )
    else:
        nx.draw_networkx_nodes(
            g, pos,
            node_size=node_size,
            node_color=node_color
        )
    if label_pos == None:
        nx.draw_networkx_labels(
            g, pos,
            labels={node: node for node in g.nodes(
            ) if node_size_dict[node] > node_label_size_limit},
            font_size=10,
            font_color='white',
            font_weight='bold'
        )
        nx.draw_networkx_labels(
            g, pos,
            labels={node: node for node in g.nodes(
            ) if node_size_dict[node] == 0},
            font_size=10,
            font_color='black',
            font_weight='bold'
        )

        x_max = max([x for x, y in pos.values()])
        y_max = max([y for x, y in pos.values()])
        x_min = min([x for x, y in pos.values()])
        y_min = min([y for x, y in pos.values()])

        y_diff = y_max - y_min
        x_diff = x_max - x_min
        hipo = (y_diff ** 2 + x_diff ** 2) ** 0.5

        max_node_size = max(node_size)
        max_node_size_canvas_ratio = 8
        font_canvas_ratio = 500
        font_margin = 12 * hipo / 2 / font_canvas_ratio

        for node in [
            node for node in nodes
            if node_size_dict[node] <= node_label_size_limit
            and node_size_dict[node] > 0
        ]:
            x, y = pos[node]
            curr_node_size = node_size_dict[node]

            default_x_margin = curr_node_size * \
                (x_diff / max_node_size / max_node_size_canvas_ratio)
            default_y_margin = curr_node_size * \
                (y_diff / max_node_size / max_node_size_canvas_ratio)

            x_margin = default_x_margin \
                if label_margin == None \
                else label_margin

            y_margin = default_y_margin \
                if label_margin == None \
                else label_margin

            x_margin_label = font_margin * (4 + len(str(node))) / 2
            y_margin_label = font_margin

            if y < 0:
                y_label_pos = y - y_margin - y_margin_label
            elif y > 0:
                y_label_pos = y + y_margin
            else:
                y_label_pos = y

            if x < 0:
                x_label_pos = x - x_margin - x_margin_label
            elif x > 0:
                x_label_pos = x + x_margin + x_margin_label
            else:
                x_label_pos = x

            plt.text(
                x_label_pos, y_label_pos, s=node,
                horizontalalignment='center'
            )
    else:
        nx.draw_networkx_labels(
            g, label_pos,
            labels={node: node for node in g.nodes()},
            font_size=10,
            font_color=label_color,
            font_weight=label_weight
        )

    if symmetric:
        if arched:
            nx.draw_networkx_edges(
                g, pos,
                edge_color=edge_color,
                width=edge_weights,
                node_size=[x + 400 for x in node_size],
                arrows=True,
                arrowsize=0,
                arrowstyle='|-|',
                connectionstyle=f'arc3,rad={arch_radius}',
            )
        else:
            nx.draw_networkx_edges(
                g, pos,
                edge_color=edge_color,
                width=edge_weights,
                node_size=[x + 400 for x in node_size],
                arrows=True,
                arrowsize=0,
                arrowstyle='|-|'
            )
    else:
        nx.draw_networkx_edges(
            g, pos,
            edge_color=edge_color,
            width=edge_weights,
            node_size=[x + 400 for x in node_size],
            connectionstyle=f'arc3,rad={arch_radius}',
        )

    if len(clusters) > 0:
        ax = plt.gca()
        ax.legend(
            handles=legend_elements,
            loc='center left', bbox_to_anchor=(1, 0.5)
        )

    plt.tight_layout()
    plt.axis('off')
    plt.show()

    if output_path != None:
        print('Saving the graph to', output_path)
        plt.savefig(output_path, format='jpg', bbox_inches='tight')

    return g