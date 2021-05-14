#ifndef GRAPH_H
#define GRAPH_H
#include <complex>
#include <cstring>
//#include <util.h>

using namespace std;

typedef vector<Vertex> Graph;

//- Construct the nearest neighbour table
void BuildGraph(Graph &NodeList, Param P){

  int q = P.q;
  int Levels = P.Levels;
  
  if(P.Vcentre == true) {

    //Level 0 spatial: trivial
    for(int mu=1; mu<q+1; mu++) {
      NodeList[0].nn[mu-1] = mu;
    }

    // CONVENTION: The link on the same level, going from node n to (n-1)
    //             is the 0th entry in the neighbour table. The next entries
    //             are the links going to the next higher level. The next
    //             entry is same level link going from n to (n+1). The last
    //             entries go to the lower level. This means all links in the 
    //             neighbour table are listed in anti-clockwise order.
    
    //Level 1
    for(long unsigned int n=endNode(0,P)+1; n<endNode(1,P)+1; n++){
      
      //This is the first node, treat it separately.
      n-1 == 0 ? NodeList[n].nn[0] = endNode(1,P) : NodeList[n].nn[0] = n-1;
      
      //Address of first new node on level l+1 from node a
      int x = endNode(1,P)+1 +(n-1)*(q-4); 
      
      //get new nodes on level 2
      for(int i=1; i<q-2; i++) {
	NodeList[n].nn[i] = x+i-1;
	NodeList[x+i-1].nn[q-1] = n;
	NodeList[x+i-1].fwdLinks = q-3;
	//By definition, the first link in this loop has two back links.
	if(i==1) {
	  NodeList[x+i-1].nn[q-2] = n;
	  NodeList[x+i-1].fwdLinks = q-4;
	  n == 1 ? NodeList[x+i-1].nn[q-1] = q : NodeList[x+i-1].nn[q-1] = n-1;
	}
      }
      NodeList[n].nn[q-2] = n%q+1;
      NodeList[n].nn[q-1] = 0;
    }
    //Fix (q-3) link on final node 
    NodeList[q].nn[q-3] = endNode(1,P)+1;
    
    //Level >=2
    for(int l=2; l<Levels+1; l++){
      
      //Loop over all nodes on this level
      //Get first new node on level l+1
      int x = endNode(l,P)+1;
      
      //Loop over all nodes on this level
      for(long unsigned int n=endNode(l-1,P)+1; n<endNode(l,P)+1; n++){      
	
	//Assign links on the same level 
	//Check if first node
	if(n == endNode(l-1,P)+1) {
	  NodeList[n].nn[0] = endNode(l,P);
	} else {
	  NodeList[n].nn[0] = n-1;
	}
	//Check if last node
	if(n == endNode(l,P)) {
	  NodeList[n].nn[NodeList[n].fwdLinks]   = n+1;
	  NodeList[n].nn[NodeList[n].fwdLinks+1] = endNode(l-1,P)+1;
	}
	else NodeList[n].nn[NodeList[n].fwdLinks+1] = n+1;
	
	//Loop over all links on node n to level l+1. If the node
	//has two back links, there are only (q-4) new links
	//to l+1, else there are (q-3).
	//By deiniftion, the first link in this loop has two back links.
	if(l<Levels) {
	  for(int i=1; i<NodeList[n].fwdLinks+1; i++) {
	    NodeList[n].nn[i] = x+i-1;
	    NodeList[x+i-1].nn[q-1] = n;
	    NodeList[x+i-1].fwdLinks = q-3;
	    if(i==1) {
	      NodeList[x+i-1].nn[q-2] = n;
	      NodeList[x+i-1].fwdLinks = q-4;
	      n == endNode(l-1,P)+1 ? NodeList[x+i-1].nn[q-1] = endNode(l-1,P) : NodeList[x+i-1].nn[q-1] = n-1;
	    }
	  }
	}
	x += NodeList[n].fwdLinks-1;
	
	//fix link q-1 on start node
	NodeList[endNode(l-1,P)+1].nn[q-1]=endNode(l-1,P);
	//fix link q-2 on start node
	NodeList[endNode(l-1,P)+1].nn[q-2]=endNode(l-2,P)+1;
	//fix link q-3 on end node
	if(n == endNode(Levels,P)) NodeList[endNode(l,P)].nn[q-3] = -1;
	else NodeList[endNode(l,P)].nn[q-3] = endNode(l,P) + 1;
      }
    }
  }
  else {  
    //Level 0
    for(long unsigned int n=0; n<3; n++){
      
      //This is the first node, treat it separately.
      n == 0 ? NodeList[n].nn[0] = 2 : NodeList[n].nn[0] =  n-1;
      
      //Address of first new node on level 1 from node n
      int x = 3 + n*(q-3);
      
      //get new nodes on level 1
      for(int i=1; i<q-1; i++) {
	NodeList[n].nn[i] = x+i-1;
	NodeList[n].fwdLinks = q-2;
	
	NodeList[x+i-1].nn[q-1] = n;
	NodeList[x+i-1].fwdLinks = q-3;
	//Corrections
	if(i==1) {
	  //This node (3,7,11) has two back links to level 0:
	  //3:  0,2
	  //7:  1,0
	  //11: 2,1
	  NodeList[x+i-1].fwdLinks = q-4;
	  NodeList[x+i-1].nn[q-2] = (x+i-1)%3;
	  NodeList[x+i-1].nn[q-1] = ((x+i-1)%3 + 2)%3;	
	}
	n == 2 ? NodeList[n].nn[q-1] = 0 : NodeList[n].nn[q-1] =  n+1;
      }
    }
    //Fix (q-2) link on final node 
    NodeList[2].nn[q-2] = 3;
    
    //Level 1
    
    //Get first new node on level 2
    int x = endNode(1,P)+1;
    
    //Loop over all nodes on level 1.
    for(long unsigned int n=endNode(0,P)+1; n<endNode(1,P)+1; n++){      
      
      //Assign links on the same level 
      //Check if first node
      if(n == endNode(0,P)+1) {
	NodeList[n].nn[0] = endNode(1,P);
      } else {
	NodeList[n].nn[0] = n-1;
      } //OK
      //Check if last node
      if(n == endNode(1,P)) {
	NodeList[n].nn[NodeList[n].fwdLinks]   = n+1;
	NodeList[n].nn[NodeList[n].fwdLinks+1] = endNode(0,P)+1;
      }
      else NodeList[n].nn[NodeList[n].fwdLinks+1] = n+1;
      
      //Loop over new links
      for(int i=1; i<NodeList[n].fwdLinks+1; i++) {
	NodeList[n].nn[i] = x+i-1;
	NodeList[x+i-1].nn[q-1] = n;
	NodeList[x+i-1].fwdLinks = q-3;
	if(i==1) {
	  NodeList[x+i-1].nn[q-2] = n;
	  NodeList[x+i-1].fwdLinks = q-4;
	  n == endNode(0,P)+1 ? NodeList[x+i-1].nn[q-1] = endNode(1,P) : NodeList[x+i-1].nn[q-1] = n-1;
	}
      }
      x += NodeList[n].fwdLinks-1;
    }
    //Fix (q-3) link on final node 
    NodeList[endNode(1,P)].nn[q-3] = endNode(1,P)+1;
    
    //Level >=2
    for(int l=2; l<Levels+1; l++){
      
      //Get first new node on level l+1
      int x = endNode(l,P)+1;    
      //Loop over all nodes on this level
      for(long unsigned int n=endNode(l-1,P)+1; n<endNode(l,P)+1; n++){      
	
	//Assign links on the same level 
	//Check if first node
	if(n == endNode(l-1,P)+1) {
	  NodeList[n].nn[0] = endNode(l,P);
	} else {
	  NodeList[n].nn[0] = n-1;
	}
	//Check if last node
	if(n == endNode(l,P)) {
	  NodeList[n].nn[NodeList[n].fwdLinks]   = n+1;
	  NodeList[n].nn[NodeList[n].fwdLinks+1] = endNode(l-1,P)+1;
	}
	else NodeList[n].nn[NodeList[n].fwdLinks+1] = n+1;
	
	//Loop over all links on node n to level l+1. If the node
	//has two back links, there are only (q-4) new links
	//to l+1, else there are (q-3).
	//By deiniftion, the first link in this loop has two back links.
	if(l<Levels) {
	  for(int i=1; i<NodeList[n].fwdLinks+1; i++) {
	    NodeList[n].nn[i] = x+i-1;
	    NodeList[x+i-1].nn[q-1] = n;
	    NodeList[x+i-1].fwdLinks = q-3;
	    if(i==1) {
	      NodeList[x+i-1].nn[q-2] = n;
	      NodeList[x+i-1].fwdLinks = q-4;
	      n == endNode(l-1,P)+1 ? NodeList[x+i-1].nn[q-1] = endNode(l-1,P) : NodeList[x+i-1].nn[q-1] = n-1;
	    }
	  }
	}
	x += NodeList[n].fwdLinks-1;
	
	//fix link q-1 on start node
	NodeList[endNode(l-1,P)+1].nn[q-1]=endNode(l-1,P);
	//fix link q-2 on start node
	NodeList[endNode(l-1,P)+1].nn[q-2]=endNode(l-2,P)+1;
	//fix link q-3 on end node
	if(n == endNode(Levels,P)) NodeList[endNode(l,P)].nn[q-3] = -1;
	else NodeList[endNode(l,P)].nn[q-3] = endNode(l,P) + 1;
      }
    }
  }    
}
#endif
