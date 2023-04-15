
RR wait time: Total wait time - part already executed - arrival time

## Indirect Communication
- Messages are directed and received from mailboxes (also referred to as a ports)
	- Each mailbox has a unique ID
	- Processes can only communicate if they share a mailbox
- Properties of a communication link
	- Processes must share a mailbox
	- A link can be associated with many processes
	- Each process can have more than one communication link
	- Some linkes are unidirectional, others are bidirectional

### Operations for Mailboxes Owned by Process
- Create a new mailbox (port)
- Send and receive messages through the mailbox
- delete a mailbox

### Operations for Mailboxes Owned by OS
- Create a new mailbox (port)
- The process that creates the mailbox initially becomes the owner and can recieve messages
- Mailbox privileges can be passed to other processes later
- Send and receive through mailbox
- Allow other processes to use a mailbox
- Delete a mailbox

## Synchronization
- Message passing can either be blocking or non blocking
- Blocking is considered **synchronous**
	- **Blocking send** the sender is blocked until the message is received
	- **Blocking receive** the receiver is blocked until a message is available

- Non blocking is considered asynchronous
	- **Non-blocking send** the sender sends the message and continues
	- **Non-blocking receive** the receiver receives 
		- A valid message, or
		- A null message
- If both send and receive are blocking we have a **rendezvous**

## Buffering
- Queue of messages attached to a link
- Implemented in one of three ways
	1. Zero capacity - no messages are queued on a link
	2. Bounded capacity - finite length
	3. Unbounded capacity - infinite length, sender never waits

## Pipes
- Acts as a conduit allowing two processes to communicate
- Ordinary pipes cannot be access from outside the process that created it
- Once the processes have finished communications and have terminated the pipe ceases to exist
- Named pipes can be accessed without a parent-child relationship


### Ordinary pipes 
- Allow communication in standard producer-consumer style
- Producer writes to one end (the write-end of the pipe)
- Consumer reads from the other end (the read-end of the pipe)
- Ordinary pipes are therefore unidirectional, if two-way communication is required, two pipes must be used, with each pipe sending data in a different direction
![[Pasted image 20230214091012.png]]

### Named Pipes
- Communication is bidirectional 
- No parent-child relationship is necessary between the communicating processes
- Several processes can use the named pipe for communication
- Continue to exist even if the communicating processes have finished
- Referred to as **FIFOs** in UNIX systems
	- A FIFO is created with mkfifo() system call and manipulated with the ordinary open(), read(), write() and close()

## Sockets
- A socket is deinfed as an endpoint for communication
- Concatenation of IP address and **port** - a number included at start of message packet to differentiate network services on a host
	- The socket 161.25.19.8:1625 refers to port 1625 on host 161.25.19.8
- All ports below 1024 are well known, used for standard services

## Remote Procedure Calls
- Remote procedure call (RPC) abstracts procedure calls between processes on networked systems
	- Again uses ports for service differentiation
- Semantics of the RPCs hides the details that allow the communication
- **Stubs** - client side proxy forthe actual procedure on the server
- The client side stub locates the server and **marshalls** the parameters

- Data representation is handled via the **External Data Representation (XDR)**
	- Big-endian and little-endian
- Messages can be delivered exactly once rather than at most once
- The OS typically provides a rendezvous or **matchmaker** services to connect client and server


