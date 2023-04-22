#CP386 

- When a thread requests resources that are not available, the thread enters the waiting state
- If the waiting state never changes, it enters **deadlock**
	- Deadlock occurs when every process in a set of processes is waiting for an event that can only be caused by another process in the set

## 8.1
- A system consists of a finate number of resources distributed among a number of competing threads
- A thread must request a resource before using it and release a resource after using it

- Under normal operation a thread follows this sequence:
	- **Request**: Requests a resource, if it can not receive it right away the thread must wait
	- **Use**: The thread can operate on the resource
	- **Release**: The thread releases the resource


## 8.2
- **Livelock** is when a thread continously attempts an action and fails
	- Real life example when two people are trying to pass each other in a hallway but move left and right and continously block each other
- Thread 1 and Thread 2 want access to Resource 1 and Resource 2
- Thread 1 locks Resource 1 and then tries to access Resource 2, Thread 2 locks Resource 2 and then tries to access Resource 1
- Both threads hold the resources they each require to proceed


## 8.3

### Necessary Conditions for Deadlock
- Deadlock can arise if the following four conditions hold simultaneously in a system:
	1. **Mutual exclusion**: at least on resource must be held in nonsharable mode; only one thread at a time can use the resource
	2. **Hold and wait**: A thread must be holding at least one resource and waiting to acquire additional resources that are currently being held by other threads
	3. **No preemption**: Resources cannot be preempted; that is, a resource can be released only voluntarily by the thread holding it, after that the thread has completed its task
	4. **Circular wait**: A set $\{T_0, T_1, ..., T_n\}$ of waiting threads must exist such that $T_0$ is waiting for a resource held by $T_1$ which is waiting for a resource held by $T_2$ and so on


### Resource Allocation Graph
- Deadlocks can be described more precisely in **system resource-allocation graphs**
- The vertices are split into two sets *T* for the active threads in a system and *R* for the resource types in the system
- A directed edge $T_i -> R_j$ is called a **request edge** and a directed edge $R_j -> T_i$ is called a **assignment edge**
![[Pasted image 20230417182103.png | 400]]
- If a resource graph contains no cycle, then no thread in the system is deadlocked
	- If the graph does contain a cycle, then a deadlock probably exists

## 8.4

- Generally a deadlock can be handled in one of three ways:
	- Ignore the problem altogether and pretend deadlocks dont exist
	- Use a protocol to prevent or avoid deadlocks, ensuring that the system will never enter a deadlocked state
	- We can allow the system to enter a deadlocked state, detect it, and recover it

- Operating systems such as Linux and Windows follow the first section, then making developers create ways to handle deadlocks 
- Other systems such as databases use the third solution to identify and recover a deadlock

- To ensure that deadlocks do not occur, the system can either us a deadlock prevention scheme or deadlock avoidance scheme

- **Deadlock prevention** provides a set of methods to ensure that at least one of the necessary conditions mentioned above (mutal exclusion, hold and wait, no preemption, circular wait) is not held, preventing a deadlock

- **Deadlock avoidance** requires that an OS is given additional information in advance concering which resources a thread will require to complete its task

- If a system does not use deadlock prevention or deadlock avoidance, then a deadlock situation may arise
- We may arise at situations where the system is in a deadlocked state but there is not way of recognizing it has happened, system performance deteriorates 

- Ignoring the possibility of deadlocks is cheaper than other approaches


## 8.5

### Mutual Exclusion
- To prevent a deadlock, mutual exlusion must hold

### Hold and Wait
- To ensure hold and wait never occurs in a system, we must guarantee that whenever a thread requests a resource it does not hold any other resources
- An alternative would be to allow a thread to request resources only when it has none

### No Preemption
- If a thread is holding some resources and requests another resource that cannot be immediately allocated, then all resources the thread is currently holding are preempted (released). 
- The thread will be restarted only when it can regain its old resources as well as the new ones it is requesting

- Alternatively, if a thread requests some resources we must check whether they are available, if they are we allocated them if not we check whether they are allocated to some other thread that is also waiting
- If so, we preempted the resources from the other thread and give it to the intial one

### Circular Wait
- Impose a total ordering of all resource types to require that each thread requests resources in an increasing order of enumeration


## 8.6
- The simplest and most useful model for deadlock avoidance requires that each thread decalre the *maximum number* of resources of each type that it may need
	- Given this information, it is possible to create an algorithm that prevents deadlock

- Linux introduced `lockdep` which prevents deadlocks, livelocks and other bugs
- When it is called, it creates a dependency graph based on the current relationships of locks and resources
- It uses the created dependency graph to identify a deadlock


- A state is *safe* if the system can allocate resources to each thread in some order and still avoid deadlock
	- This is called a **safe sequence**
- If no such sequence exists, then the system state is *unsafe*
	- An *unsafe* state may lead to a deadlock

### Resource-Allocation-Graph-Algorithm
- A deadlock detection algorithm used in OS
- To detect cycle a RAG uses the following appraoch
	- Start with a RAG containing all vertices and edges
	- Find a vertex that has no outgoing request edges
	- If there is no such vertex, the system is deadlocked
	- Remove the vertex and its incoming assignment edge from the RAG
	- Repeat steps 2-4 until all vertices have been remove
	- If any vertices remain, the system is deadlocked

- There is a new type of edge called a **claim edge** where a Thread may request a Resource at some time in the future
	- This is represented by a dotted line

### The Bankers Algorithm
- The bankers algorithm is used to avoid deadlocks in a system where multiple processes compete for a finite number of resources
	- It works by keeping track of the available resources, the maximum demand of each process and the number of resources currently allocated to each process

- The algorithm is as follows:
	- At the start, the system has a fixed number of resources available
	- Each process in the system specifies the maximum number of resources it may need to complete its task
	- When a process requests resources, the Banker's algorithm checks whether the request can be granted without getting a deadlock
	- If the request can be granted, the resources are allocated, the system then checks if the new state is safe
	- If the new state is safe, the process can continue and the Banker's Algorithm updates the available resources accordingly
	- If the request cannot be granted, the process is blocked until the required resoruces become available



## 8.7
- If a system does not have deadlock prevention or deadlock avoidance, it may provide
	- An algorithm that eximes the state of the system to determine whether or not a deadlock has occured
	- An algorithm to recover from a deadlock


### Single Instance of Each Resource Type
- If all resources have only a single instance, then we can define a deadlock detection algorithm that uses a variant of the resource-allocation graph called a **wait-for** graph
- We obtain this graph from the resource-allocation graph by removing the resource nodes and collapsing the appropriate edges
![[Pasted image 20230417192943.png | 400]]
- A deadlock exists only if the wait-for graph contains a cycle
- To detect deadlocks, the system needs to *maintain* the waiting for graph and periodically *invoke an algorithm* that searches for a cycle 


## 8.8
- One possibility to recover from a deadlock is to let the system **recover** from the deadlock automatically
- There are two options for breaking a deadlock
	- One is to simply abort one or more threads to break the circular wait
	- The other is to preempt some resources from one or more of the deadlocked threads


#### Process and Thread Termination
- The eliminate deadlocks by aborting a process or thread, we use one of two methods
	- **Abort all deadlocked processes**: This method breaks deadlocks but at great expense
	- **Abort one process at a time until the deadlock cycle is eliminated**: This method has considerable overhead because the deadlock detection algorithm must run each time a process is eliminated to determine if there is still a deadlock


#### Resource Preemption
- To eliminate deadlocks using resource preemption, we successively preempt some resource from processes and give these resources to other processes until the deadlock cycle is broken
- If the preemption is to break a deadlock, three issues must be addressed:
	- **Selecting a victim**: Must determine which process to end and which has the minimum cost of ending
	- **Rollback**: Rollback the process to the very beginning? or just before the deadlock? must have information on the state of all running processes
	- **Starvation**: How do we ensure starvation of other processes will not occur?