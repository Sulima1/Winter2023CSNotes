#CP386 

*Review Diagram of Interrupt Process slides for exam*

## CPU Schedulers
- Short term scheduler 
	- Selects which processes should be executed next
- Long term scheduler
	- Selects which processes should be brought into ready queue
	- Controls the degree of **multiprogramming**
		- Degree of multiprogramming is the number of processes that are currently in main memory
	- Strives for good **process mix**


**a process must be in main memory for its execution**

## Medium Term Scheduling
- Can be added if the degree of multiprogramming needs to increase
	- Remove a process from memory, store on disk and bring it back in from disck to continue execution
	- Typically only necessary when memory has been overcommited and must be freed up


## Context Switching
- When CPU switches to another process, the system must save the state of the old process and load the saved state of the next process via a **context switch**


## Process Creation
- Parent process create children processes which in turn create other processses, forming a tree of processes
- Generally, process identified and mangaed via the **process identifier (pid)**

- Resource sharing options
	- Parent and children share all resources
	- Children share subset of parents resources
	- Parent and child share no resources
- Execution options
	- Parent and children execute concurrently 
	- Parent waits until children terminate

- UNIX examples:
	- fork() system call creates new process - *important*
	- exec() system call used after fork() to replace the process' memory space with a new program

## The fork() system call
- fork() creates a new process
- The child is a *copy* of the parent, but... 
	- It has a different pid (thus ppid)
	- Its resource utilization so far has been set to 0
- fork() returns the childs pid to the parent and 0 to the child
	- Each process can find its own pid in the getpid() call 
- Both processes continue execution after the call to fork()

```c
pid_t pid = fork();
if (pid == 0){
	printf("child process created")
}
else if (pid < 0){
	fprintf(stdout, "cant fork erorr");
}
```

- *fork() calls create children processes that run through the rest of the code exponentially*
	- If there are n fork() calls, there will be 2^n children processes running through the code