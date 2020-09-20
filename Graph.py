# -----------------------------------------------------------
# Transportation network Class 
# -----------------------------------------------------------

import osmnx as ox
import networkx as nx
import numpy as np 
import pandas as pd


class Graph:
    """
    Class for transportation network and counters data

    """
    def __init__(self,network,date):
         
        self.network = network
        self.adj= nx.adjacency_matrix(network)
        self.date = date
        self.counts=None 
        self.paths={}
        self.indices=None
        self.weights=None
        self.OD=None
    def get_counts(self,counts,positions):
        """
         Assign traffic counts to edges of the network
         counts: dataframe of traffic counts with columns date, and debit
         positions: dataframe with latitude and longitude of counters
        """
        count_dict= {}
        for row in positions.itertuples():
            try:
                val = counts.loc[(counts.edge_id==row.edge_id)&(counts.date==date),"debit"].values[0]
               
                if not np.isnan(val):
                    ne = ox.get_nearest_edge(self.network, (row.lat,row.lon))
                    count_dict[tuple(ne[:2])]= val
            except:
                continue
        self.counts=count_dict
   
    def sample_counts(self,od=None,nb=100):
         """
         Sample weights from gamma distribution and generate traffic on edges
        """
        adj=self.adj.toarray()
        K = np.shape(adj)[0]
        
        if od is None:
            ind = np.random.randint(K,size=nb)
            w = np.random.gamma(1,1, size=nb).reshape(nb,1)
            lam = w.dot(w.transpose())
            n = np.random.poisson(lam).flatten()
            comb = list(itertools.product(ind, ind))
            orig= [a[0] for a in comb]
            dest= [a[1] for a in comb]
            od = sps.csr_matrix((n, (orig,dest)), shape=(K, K)).toarray()
            np.fill_diagonal(od, 0)
        else: 
            ind= np.random.randint(K,size=od.shape[0])
            
            
        counts={}
        
        for i, j in zip(*od.nonzero()):
            o=np.array(list(self.network))[i]
            d=np.array(list(self.network))[j]
            
            path=nx.shortest_path(self.network,o,d)
            self.paths[(o,d)]=path
            
            for pth in zip(path, path[1:]):
                if pth in counts.keys():
                    counts[pth]=counts[pth]+od[i,j]
                else:
                    counts[pth]= od[i,j]
                
        self.counts=counts
        self.indices=ind
        self.weights=w
        self.OD=od
    
    def get_path(self, O, D):
        """
         returns a list with shortest path between O and D
        """
        if (O,D) in self.paths:
            return self.paths[(O,D)]
            
        try:
            path=nx.shortest_path(self.network,O,D,weight="length")
            self.paths[(O,D)]=path
        except:
            path=[]
        return path
    
def counts_in_path(G, path):
        """
          returns a dictionary with the traffic on the path 
        """
    if len(path)< 2:
        return {}
    try:
        counted_path= {k : v for k,v in  G.counts.items() if k in zip(path, path[1:])}

        return counted_path
    except:
        print("Counts are not assigned")

    return counted_path
        
        