import networkx as nx
import matplotlib.pyplot as plt

class BucketElimination():
    
    """
    A class that implements the basic methods for the bucket elimination in a tabular form


    Attributes
    ----------
        domain : list
            the domain of the variables, a dictionary with the variable name as key and a list for the discrete domain as a value.
        problem_variables_ordered : list
            the variables that appear in the problem as a list of literals, ordered following the add chain.
        problem_soft_constraints : list
            the soft contraints, a list of lists, each list is built with the function name for the first element, 
            followed by the intereseted variables.
        problem_hard_constraints : int
            the hard contraints (only inequality constraints), a list of lists, 
            each list represent the variable interested in the inequality contraints.

    Methods
    -------
        add( bucket )
            method that add an object of the class bucket to the problem. 
        bucket_processing()
            process all the bucket in the given order (following the add chain)
        value_propagation()
            propagate the value obtained with the bucket elimination to obtain the global maximum of the given problem 
            and the corresponding assignment for the variables.
        plot_assignment_as_graph( assignment, soft_eval )
            plot the colored graph following the assignment for the variables.
        get_tables()
            get method that returns the list of the generated tables
        
        
    """
        
    
    def __init__( self, domain ):
        
        """
        Constructor of the class

        Parameters
        ----------
        domain : list
            the domain of the variables, a dictionary with the variable name as key and a list for the discrete domain as a value.
        """
        
        # Public variables
        self.domain = domain
        self.problem_variables_ordered = []
        self.problem_soft_constraints = []
        self.problem_hard_constraints = []
        
        # Private variables
        self._bucket_list = []
        self._functions = []
        self._h_functions = []
        self._assignment = {}
        
        
    def add( self, bucket ):
        
        """
        Method that add an object of the class bucket to the problem. 
        The method also extract the interested variables and update the domains of the problem.

        Parameters
        ----------
            bucket : Bucket 
                the bucket to add to the problem
        """
        
        #
        self.problem_variables_ordered.insert( 0, bucket.variable )
        self.problem_soft_constraints += bucket.soft_cnst
        self.problem_hard_constraints += bucket.ineq_cnst
        
        self._bucket_list.append( bucket )
        
    
    def bucket_processing( self ):
        
        """
        Process all the bucket in the given order (following the add chain), generating all the tables and h_tables.
        The method also store the generated tables inside the variable "functions".
        
        """
            
        # Iterate over all the buckets
        for bucket in self._bucket_list:
            
            # Find the previously computed h_function and add the variables from the h fucntion  
            h_variables = []
            h_functions = []
            to_delete_h_functions = []
            for h_entry in self._h_functions:
                if bucket.variable in h_entry["Domain"]:
                    h_functions.append(h_entry["Table"])
                    to_delete_h_functions.append(h_entry)
                    h_variables += [var for var in h_entry["Domain"]]
                    
            # Delete duplicate from list                   
            h_variables = sorted(list(set(h_variables)))
                        
            # Compute table and h function
            table, h_table, used_variables = bucket.get_tables( self.domain, h_functions, h_variables )
            
            # Store all the generated tables
            self._functions.append( table.copy() )
            
            # Store the old h_functions
            self._h_functions.append({})
            self._h_functions[-1]["Name"] = f"h_{bucket.variable}"
            self._h_functions[-1]["Domain"] = [var for var in used_variables if var != bucket.variable]
            self._h_functions[-1]["Table"] = h_table
            
            # Remove the processed h function
            for h_fnc in to_delete_h_functions: self._h_functions.remove(h_fnc)
            
            
    def value_propagation( self ):
            
        """
        Propagate the value obtained with the bucket elimination to obtain the global maximum of the given problem and the corresponding assignment for the variables.

        Returns:
        --------
            assignment : dict
                the assignment for each variable to obtain the maximum (the key is the literal and the value is the assigned value) 
            global_maximum : int
                the value of the global maximum for the soft cosnstraints that respect all the hard (inequality) constraints
                
        Raises
        ------
            ValueError
                If the table is empty, no valid solution exists that respects the hard constraints
        """
            
        assignment = {}
        global_maximum = None

        # Iterate in the given order to compute maaximum and best assignment
        for bucket_table in reversed( self._functions ):

            # Get the sum of the already with the already assigned variables
            candidate = []
            for row in bucket_table:
                condition_array = [row[key] == value for key, value in zip(assignment.keys(), assignment.values()) if key in list(row.keys())]
                if all(condition_array): candidate.append(row)

            # Check if there are candidate for this assignment
            if len(candidate) == 0: raise ValueError("No solution that respect all the hard constraints exists")

            # Update the assignment list
            maximum = max([el['SUM'] for el in bucket_table])
            for row in candidate:
                if row['SUM'] == maximum: 
                    for key in row.keys():
                        if key in self.problem_variables_ordered: assignment[key] = row[key]
                    break  

            # Update the global_maximum (just the first time)
            global_maximum = maximum if global_maximum == None else global_maximum

        #
        return assignment, global_maximum
    
    
    def plot_assignment_as_graph( self, assignment, soft_eval ):
    
        """
        Plot the colored graph following the assignment for the variables.
        
        Parameters
        ----------
            assignment : list 
                the assigned value for each variable.
            variables : list
                the variables name for the problem, a list of string with the name of the variables.

        Raises
        ------
            ValueError
                If the assigned color to one variable is unknown (i.e., not present inside the color map dictionary)   
        """

        # Define the color map that translate the assigned value
        # in a matplot lib understandable string
        color_map = {'R':'red', 'B':'royalblue', 'G':'green', 'Y':'yellow'}
        for val in assignment.values():
            if not val in color_map.keys(): raise ValueError("The color of the assignment is unknown (add the key in the color map) ")

        # Create the graph object and assign node and edges variables
        G = nx.Graph()
        G.add_nodes_from(self.problem_variables_ordered)
        G.add_edges_from(self.problem_hard_constraints)
        G.add_edges_from( [cst[1], cst[2]] for cst in self.problem_soft_constraints)

        # Create the arrays with the color and labels for the edges
        node_color = [color_map[assignment[var]] for var in self.problem_variables_ordered]
        edge_labels_hard= {(cst[0], cst[1]):'hard' for cst in self.problem_hard_constraints}
        # EDIT: removed commented row 207 in actual row 210 
        # edge_labels_soft = {(cst[1], cst[2]):f"soft (val={soft_eval[idx]})" for idx, cst in enumerate(self.problem_soft_constraints)} 
        # NB: after fix, main problem change:
        #     => bucket_elimination.plot_assignment_as_graph( assignment, [[evaluations[idx], cst[1], cst[2]] for idx, cst in enumerate(problem_soft_constraints)] )
        edge_labels_soft = {(cst[1], cst[2]): f"soft (val={cst[0]})" for cst in soft_eval} 
        edge_labels_soft.update(edge_labels_hard)

        # Call networkx and matplotlib to plot the graph
        pos = nx.spring_layout(G)
        nx.draw(G, pos, node_size=1000, node_color=node_color, width=2, with_labels=True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_soft, font_size=15)
        plt.show() 
        
        
    ###################
    #   GET METHODS   #
    ###################
    
    
    def get_tables( self ): return self._functions
        