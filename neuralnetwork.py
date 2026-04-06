import random
import numpy as np
from .config import gates, GATE_KEYS
import networkx as nx
import matplotlib.pyplot as plt

class BinaryNeuralNetwork:
    def __init__(self, callbacks):
        self.node_counter = 0
        self.nodes, self.connections = [], []
        self.callbacks = callbacks # this should contain the "find_or_create_innovation" method
        self.initialize(callbacks['config'].getint('BinaryNeuralNetwork', 'num_inputs'), callbacks['config'].getint('BinaryNeuralNetwork', 'num_outputs'))

    def next_node_id(self):
        self.node_counter += 1
        return self.node_counter

    def initialize(self, num_input, num_output):
        self.nodes = [Node(self.next_node_id(), node_type='input', callbacks=self.callbacks) for _ in range(num_input)]
        self.nodes += [Node(self.next_node_id(), node_type='output', callbacks=self.callbacks) for _ in range(num_output)]

        # connect some inputs to outputs
        input_nodes = [n for n in self.nodes if n.node_type == 'input']
        output_nodes = [n for n in self.nodes if n.node_type == 'output']

        for out_node in output_nodes:
            if len(input_nodes) >= 2:
                selected_inputs = random.sample(input_nodes, 2)
                self.add_connection(selected_inputs, out_node)

    def feed_forward(self, inputs):
        input_nodes = [n for n in self.nodes if n.node_type == 'input']
        if len(inputs) != len(input_nodes):
            raise ValueError(f'Expected {len(input_nodes)} inputs, got {len(inputs)}')

        # Reset all nodes
        for node in self.nodes:
            node.value = 0.0
            node.ready = False

        # Set input values
        for node, val in zip(input_nodes, inputs):
            node.value = val
            node.ready = True

        # Process hidden and output nodes
        while not all(node.ready for node in self.nodes):
            for node in self.nodes:
                if node.ready:
                    continue

                incoming = [c for c in self.connections if c.out_node == node and c.enabled]
                if not incoming:
                    node.ready = True
                    continue
                if len(incoming) > 1:
                    raise RuntimeError(f'Node {node.id} has multiple incoming connections')
                conn = incoming[0]

                in1, in2 = conn.in_nodes[0], conn.in_nodes[1]
                if not (in1.ready and in2.ready):
                    continue

                node.value = node.compute(in1.value, in2.value)
                node.ready = True

        return [n.value for n in self.nodes if n.node_type == 'output']

    def add_random_node(self):
        '''
        Inserts a new node onto a random enabled connection, splitting it.
        Original connection is disabled, replaced by:
          (original inputs) -> new_node -> (original output)
        '''
        enabled_conns = [c for c in self.connections if c.enabled]
        if not enabled_conns:
            return

        conn = random.choice(enabled_conns)  # use random, not np.random
        in_nodes = conn.in_nodes   # list of input nodes
        out_node = conn.out_node

        # Disable the original connection
        conn.enabled = False

        # Create new hidden node
        new_node = Node(self.next_node_id(), node_type='hidden', callbacks=self.callbacks)
        self.nodes.append(new_node)

        # Create first new connection: (original inputs) -> new_node
        try:
            self.add_connection(in_nodes, new_node)
        except ValueError:
            # If this fails, roll back? For simplicity, just return.
            conn.enabled = True   # re-enable original
            self.nodes.remove(new_node)
            return

        # Create second new connection: new_node -> original output
        try:
            self.add_connection([new_node, new_node], out_node)   # single input list
        except ValueError:
            # Rollback: remove the first connection and the node
            # Find and remove the connection we just added (by matching in_nodes & out_node)
            for c in self.connections:
                if set(c.in_nodes) == set(in_nodes) and c.out_node == new_node:
                    self.connections.remove(c)
                    break
            self.nodes.remove(new_node)
            conn.enabled = True
            return
            
    def modify_random_connection(self):
        enabled_conns = [c for c in self.connections if c.enabled]
        if not enabled_conns:
            return
        conn = random.choice(enabled_conns)
        out_node = conn.out_node
        possible_in = [n for n in self.nodes if n.node_type != 'output' and n != out_node]
        if len(possible_in) < len(conn.in_nodes):
            return

        retries = 0
        max_retries = 10
        while retries < max_retries:
            new_in_nodes = random.sample(possible_in, len(conn.in_nodes))
            if set(new_in_nodes) == set(conn.in_nodes):
                retries += 1
                continue
            try:
                # Disable the old connection and add a new one with the same innovation?
                # For simplicity, we just add a new connection (might create duplicate innovation numbers)
                # Better: modify in place, but if you have validation, do:
                conn.innovation_number = self.validate_connection(new_in_nodes, out_node)
                conn.in_nodes = new_in_nodes
                break
            except ValueError:
                retries += 1

    def swap_random_input_nodes(self):
        """
        Randomly selects an enabled connection and swaps the order of its input nodes.
        """
        enabled_conns = [c for c in self.connections if c.enabled]
        if not enabled_conns:
            return  # nothing to swap
        conn = random.choice(enabled_conns)
        conn.swap_input_order()

    def change_random_gate(self):
        '''
        changes the activation function of a random node
        '''
        eligible = [n for n in self.nodes if n.node_type != 'input']
        node = random.choice(eligible)
        node.gate = random.choice(GATE_KEYS)

    def add_connection(self, in_nodes, out_node):
        if len(in_nodes) != 2:
            raise ValueError('Every connection must have exactly 2 input nodes')

        # Point 6 fix: enforce exactly one incoming connection per non-input node
        if any(c.out_node == out_node and c.enabled for c in self.connections):
            raise ValueError(f'Node {out_node.id} already has an incoming connection')

        # Check if connection already exists (using set for order-independence)
        if any(set(c.in_nodes) == set(in_nodes) and c.out_node == out_node for c in self.connections):
            raise ValueError('Connection already exists')
    
        # Check if connection creates cycles
        if self.would_create_cycle(in_nodes, out_node):
            raise ValueError('Connection would create a cycle')
    
        # Create the connection
        innovation_number = self.callbacks['find_or_create_innovation'](in_nodes, out_node)
        self.connections.append(Connection(innovation_number, in_nodes, out_node))
        
 
    def validate_connection(self, in_nodes, out_node):
        if len(in_nodes) != 2:
            raise ValueError('Every connection must have exactly 2 input nodes')
        # Check if connection already exists
        if any(set(c.in_nodes) == set(in_nodes) and c.out_node == out_node for c in self.connections):
            raise ValueError('Connection already exists')
    
        # Check if connection creates cycles
        if self.would_create_cycle(in_nodes, out_node):
            raise ValueError('Connection would create a cycle')
    
        # Create the connection
        return self.callbacks['find_or_create_innovation'](in_nodes, out_node)

    def would_create_cycle(self, in_nodes, out_node):
        '''
        Checks if adding connections from a list of input nodes to a single output node would create any cycle
        '''
        # Create a directed graph from the connections
        G = nx.DiGraph()
    
        # Add each enabled connection as a directed edge: from in_node to out_node
        for connection in self.connections:
            if connection.enabled:
                # connection.in_nodes is a list (e.g., [nodeA, nodeB])
                # We need to add an edge from each input node to the output node
                for in_node in connection.in_nodes:
                    G.add_edge(in_node.id, connection.out_node.id)
        
        # Now test the potential new connections: add edges from each in_node to out_node
        for in_node in in_nodes:
            G.add_edge(in_node.id, out_node.id)
    
        # Check if the graph has a cycle
        try:
            nx.find_cycle(G)
            return True
        except nx.exception.NetworkXNoCycle:   
            return False

    def visualize(self):
        G = nx.DiGraph()

        # Add nodes to the graph
        for node in self.nodes:
            if node.node_type == 'input':
                G.add_node(node.id, pos=(0, node.id))
            elif node.node_type == 'output':
                G.add_node(node.id, pos=(1, node.id))
            else:  # hidden nodes
                # Generate random position for hidden nodes
                x = random.uniform(0.2, 0.8)
                y = random.uniform(0, 3)
                G.add_node(node.id, pos=(x, y))

        # Add edges to the graph
        for connection in self.connections:
            if connection.enabled:
                # Iterate over all input nodes in the connection's list
                for in_node in connection.in_nodes:
                    G.add_edge(in_node.id, connection.out_node.id, weight=connection.weight)

        # Get the positions of the nodes
        pos = nx.get_node_attributes(G, 'pos')

        # Draw the nodes
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')

        # Draw the edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)

        # Draw the node labels (node IDs)
        node_labels = {node.id: str(node.id) for node in self.nodes}
        nx.draw_networkx_labels(G, pos, labels=node_labels)

        plt.axis('off')
        plt.show()


class Node:
    def __init__(self, id=None, node_type='hidden', callbacks=None):
        if id is None:
            raise ValueError('Node must have an id')

        self.id = id
        self.node_type = node_type
        self.callbacks = callbacks
        self.gate = random.choice(GATE_KEYS)
        self.value = 0.0
        self.ready = False
        
    def compute(self, val1, val2):
        """
        Implements the gates on the input
        """
        return gates[self.gate](val1, val2)

class Connection:
    def __init__(self, innovation_number, in_nodes, out_node, enabled=True):
        self.in_nodes = in_nodes
        self.out_node = out_node
        self.innovation_number = innovation_number
        self.enabled = enabled

        # Check each input node (source) for validity
        for in_node in self.in_nodes:
            if in_node.node_type == 'output':
                raise ValueError('Output nodes cannot have outgoing connections')
            if in_node == self.out_node:
                raise ValueError('Connection cannot be made between the same node')

        # Check the output node (target)
        if self.out_node.node_type == 'input':
            raise ValueError('Input nodes cannot have incoming connections')
            

    def swap_input_order(self):
        """
        Swaps the order of input nodes in this connection.
        For a connection with exactly 2 inputs, this exchanges them.
        For connections with more than 2 inputs, swaps two randomly chosen positions.
        """
        if len(self.in_nodes) == 2:
                self.in_nodes[0], self.in_nodes[1] = self.in_nodes[1], self.in_nodes[0]

