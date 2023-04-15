#CP386 

![[Pasted image 20230207084836.png]]

![[Pasted image 20230207084846.png]]


## The exec*() Family of system calls

- The exec system call replaces the process image by that of a specific program
- Essentially one can specify:
	- Path for the executable
	- Command-line arguments to be passed to the executable
	- Possibly a set of environment variables

![[Pasted image 20230207085023.png]]

### Other exec calls avaiable in <unistd.h> library
- execl()
- execlp()
- execle()
- execv()
- execvp()
- execve()


### Understanding the syntax of exec() calls

```c
int execvp (const char *file, char *const argv[])
```
- File points to the executable file name associated with the file being executed
- argv is a NULL-terminated array of character pointers that contain the executables information


## wait() and waitpid()

- A parent can wait for a child to complete
- The wait() call
	- blocks until any child completes
	- returns the pid of the completed child and the child's exit cost
- The waitpid() call
	- blocks until a specific chil completes
	- can be made non-blocking

![[Pasted image 20230207090823.png]]


## Process Termination
- Process executes last statement and then asks the operating system to delete it using the exit() system call
	- Returns status data from child to parent (via wait())
	- Process resources are deallocated by operating system
- Parent may terminate the execution of children processes using the abort() system call. Some reasons for doing so:
	- Child has exceeded allocated resources
	- Task assigned to child is no longer required
	- The parent is exiting and the operating system does not allow a child to continue if its parent terminates

- Some operating systems do not allow child to exist if tis parent has been terminated. If a process terminates, then all its children must be terminated
	- **Cascading termination**, All children, grandhildren etc. are terminated
	- The termination is initiated by the operating system
- The parent process may wait for termination of a child process using the wait() system call
	- The call returns status information and the pid of the terminated process
```c
pid = wait(&status);
```

## Zombie Process
- If no parent is waiting (did not invoke wait()), the process is a **zombie**

- A zombie process is any process that has finished executing, but entry to the process is available in the process table for reporting to the parent process
- A process table is a data structure that stores all the process-related information in an operating system that includes the process' exit status
- A process that is not removed from the process table is considered a zombie
	- The parent process removes the process entry with the exit status of the child process 


## Orphan Process
- If parent is terminated without invoking wait(), the process is an **orphan**

- A process that does not have a parent process is an orphan process. A child process becomes an orphan when either of the following occurs
	- When the task of the parent process finishes an terminates without terminating the child process
	- When an abnormal termination occurs in the parent process

## Multiprocess Architecture - Chrome Browser
- **Browser**: process manages user interface, disk and network I/O
- **Renderer**: Process renders web pages, deals with HTML, Javascript, A new renderer created for each website opened
	- Runs in a **sandbox** restricting disk and network I/O. Minimizing effect for security exploits
- **Plug-in** process for each type of plug-in


## Interprocess Communication
- Processes within a system may be **independent** or **cooperating**
- **Independent** process cannot affect or be affected by the execution of another process
- **Cooperating** process can affect or be affected by the execution of another process

- Cooperating proceses need **interprocess communication (IPC)**
- Two models of IPC
	- **Shared memory**
	- **Message passing**
- Shared memory is faster than MP as MP is implemented using system calls which is a time consuming task
- In shared memory system calls are only required to establish a shared memory region. Once setup all access is treated as routine memory access
- MP is easier to implement because no conflict needs to be avoided

![[Pasted image 20230207115216.png]]

- Typically, a shared-memory region resides in the address space of the process creating the shared-memory segment


## Producer - Consumer Problem
- Producer process produces information that is consumed by a consumer process
	- **Unbounded-buffer**: places no practical limit on the size of the buffer. Consumer may have to wait()
	- **Bounded-buffer** assumes that there is a fixed buffer size. Consumer may have to wait if it is empy and producer may have to wait if it is full

## Bounded Buffer Shared Memory Solution
```c
#define BUFFER_SIZE 10
typedef struct {
  ....
} item;

item buffer[BUFFER_SIZE];
int in = 0;
int out = 0;
```

- The variable in points to the next free position in the buffer
- Out points to the first full position in the buffer
- The buffer is empty when in == out
- The buffer is full when ((in + 1) % BUFFER_SIZE) == out


## Bounded Buffer Producer
```c
item next_produced;
while (true){
	// produce an item in next produced
	while (((in + 1 ) % BUFFER_SIZE) == out)
		// do nothing
	buffer[in] = next_produced;
	in = (in + 1) % BUFFER_SIZE;
}
```

## Bounded Buffer Consumer
```c
item next_consumed;

while (true){
	while (in==out)
		// do nothing
	next_consumed = buffer[out];
	out = (out + 1) % BUFFER_SIZE;
//consume the item in next consumed
}
```


## Interprocess Communication - Message Passing
- The mechanism for processes to communicate and synchronize their actions 
- Message system - processes communicate with each other without resorting to shared variables
- IPC facility provides two operations
	- send(message)
	- receive(message)
- The message size is either fixed or variable
- Implementing the variable size message are complex but makes programmers lives easier

- If processes P and Q wish to communicate, they need to:
	- Establish a **communication link** between them
	- Exchange messages via send/recieve
- Implementation issues:
	- How are links established?
	- Can a link be associated with more than two processes?
	- How many links can there be between every pair of communicating processes
	- What is the capacity of a link
	- Is the size of a message that the link can accomade fixed or variable
	- Is a link unidirectional or bi-directional

### Implementation of a Communication Link
- Physical:
	- Shared memory
	- Hardware bus
	- Network
- Logical:
	- Direct or indirect
	- Synchronous or asynchronous
	- Automatic or explicit buffering

## Direct Communication
- Processes must name each other explicitly
	- **send** (P, message) - send a message to process P
	- **receive**(Q, message) - receive a message from process Q
- Properties of communication link:
	- Links are established automaticlaly
	- A link is associated with exactly one pair of communicating processes
	- Between each pair there exists exactly one link
	- The link may be unidirectional but is usually bi-directional

- This scheme exhibits symmetry in addressing; that is, both the receiver and sender process must name the other to communicate
- A variant of this scheme employs asymmetry in addressing
	- In asymmetry, the sender names the recipient; the recipient is not required to name the sender
		- send(P, message) - send a message to process P
		- receive(pid, message) - receive a message from any process

