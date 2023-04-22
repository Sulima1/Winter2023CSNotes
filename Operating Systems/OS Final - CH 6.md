#CP386 

## 6.1

- A **cooperating process** is one that can be affected by other processes executing in the system
	- Can either directly share logical address space or be allowed to share data through shared memory

- Concurrent or parallel execution can contribute to data integrey issues when sharing with multiple processes


- This code adds a integer variable count, *count*, and increments *count* each time an item is added (decrements each time one is removed)
INCREMENT
```c
while (true) {  
	/* produce an item in next produced */  
	while (count == BUFFER SIZE)  
		; /* do nothing */  
	buffer[in] = next produced;  
	in = (in + 1) % BUFFER SIZE;  
	count++;  
}
```
DECREMENT
```c
while (true) {  
	while (count == 0)  
		; /* do nothing */  
	next consumed = buffer[out];  
	out = (out + 1) % BUFFER SIZE;  
	count--;  
	/* consume the item in next consumed */  
}
```

- The issue with these codes is that *count*, if executed concurrently will sometimes increment or decrement incorrectly (because the two processes are sharing the same *count* in memory and not locking)
	- This is called a **race condition**


## 6.2

- A process has a segment of code called a **critical section** where the processes is accessing or updating data that is shared among other processes
	- When one process is in its critical section, it is important that no other processes are allowed to execute in the critical section
	- **critical section problem**

- Each process must request access to a critical section in the **entry section** and end with an **exit section**
	- Any of the remaining code is the **remainder section**
![[Pasted image 20230416152321.png | 350]]

- A solution to the critical-section problem must solve three requirements
	- **Mutual exclusion**: If a process is executing its critical section, no other processes can execute their critical sections
	- **Progress**: If no process is executing its critical section and a process wishes to enter its critical section, then only the processes that are *not executing their remainder sections* can participate in deciding which process can enter its critical section
	- **Bounded waiting**: There exists a bound, or limit on the number of times that the other processes are allowed to enter their critical sections after a process has made a request to enter its critical section and before that request is granted

- Code being run at the operating system level is called **kernel code**
- Kernel code is subject to many possible race conditions
	- The next available PID being assigned to 2 processes unless mutual exclusion is applied

- Message passing is not a viable solution to the critical section in a multiprocessor system, message passing takes time to go between processors

- There are two general approaches to handle critical sections
	- **preemptive kernels**
	- **nonpreemptive kernels**

- A **preemptive kernel** allows a process to be preempted while it is running in kernel mode
- A **nonpreemptive** does not allow a process to be preempted
- When code is preempted in kernel mode it is when the kernel interrupts one process to begin another

- Preemptive kernels are especially difficult to design for SMP architectures
	- SMP is symmetric multiprocessing, computers with multiple processors that share the same memory and I/I subsystems

- Why would anyone have a preemptive kernel over a nonpreemptive one?
	- A preemptive kernel is more responsive (less risk for a kernel mode process to run for too long)
	- A preemptive kernel is more suitable for real-time programming


## 6.3

- **Peterson's Solution**: solution to the critical section problem
- j = 1 - i
```c
int turn;  
boolean flag[2];

while (true) {  
	flag[i] = true;  
	turn = j;  
	while (flag[j] && turn == j)  
		;  
		/* critical section */  
	flag[i] = false;  
		/*remainder section */  
}
```
- The variable *turn* indicates whose turn it is to enter its critical section
- the *flag* array is used to indicate if a process is ready to enter its critical section

- The algorithm is **mutually exclusive** because each process only enters its critical section is `flag[j] == false` or `turn == i`

- The process shows **progress** and **bounded waiting** because a process can be prevented from entering the critical section only if it is stuck in the while loop
- Since one process does not change the *turn* variable while executing, the process will enter its critical section (progress) after at most one entry by the one process (bounded waiting)

- Peterson's solution is not guaranteed to work on modern computers
	- New processors need to reorder read and write operations that have no dependencies


## 6.4

- Three hardware solutions that provide support for solving critical section problem

### Memory Barriers
- **Memory model** is program that determines what applications get memory
- Falls into one of two categories
	- **Strongly ordered**: a memory modification on one processor is immediately visible to all other processors
	- **Weakly ordered**: modifications to memory on one processor may not be immediately available to other processors

- **Memory barriers/fences** are instructions that can force changes in memory to be propagated through other processors


### Hardware Instructions
- Modern computers allow us to swap the contents of two words **automically**, meaning as an uninterruptible unit
- If two function calls are executed simultaneously, they they will be executed sequentially in some arbitrary order

- The common datastructures are 
```c
boolean waiting[n];
int lock;
```

### Atomic Variables

- Something is atomic if it is executed as a single, indivisible step without the possibility of being interrupted

- **Atomic variables**: provide atomic operations on basic data types like integers and booleans
- Atomic variables do not entirely avoid race conditions in all cases


## 6.5

- mutex in **mutex lock** stands for **mu**tual **ex**clusion
- Protects critical sections and prevents race conditions
- `acquire()` acquires a lock and `release()` releases the lock
- mutex has a boolean varaible `available` that indicates if the lock is available or not
		- `acquire()` sets available to false, `release()` sets available to true
![[Pasted image 20230416223449.png | 350]]

- Locks are either **contended** or **uncontended**
	- A lock is **contended** if a thread blocks while trying to acquire a lock
	- It is **uncontended** if a lock is available when a thread attempts to acquire it
- Contended locks experience **high contention** or **low contention** depending on the number of threads trying to access it
	- Highly contended locks *decrease* performance of concurrent applications

- **Spinlocks** are a method that attempts to access a lock by spinning in a loop repeatedly requesting a lock until it becomes available
	- Type of mutex lock

	- The main disadvantage of the `acquire()` and `release()` is that it introduces **busy waiting**
		- While a process is locked, any other process that attempts to run its critical section must continously loop until it can acquire the lock
		- This is an issue


## 6.6

- A **semaphore** is an integer variable that is only accessed through two atomic operations: `wait()` and `signal()`
	- `wait()` decrements the value of the semaphore and `signal()` increments the value of the semaphore

- There are two types of semaphores **counting semaphores** and **binary semaphores**
	- **counting semaphores** can range over a number of values, **binary semaphores** can only equal 0 or 1
- Counting semaphores are used to control access to a resource with a finite number of instances
	- It is initialized to the number of resources available

- Semaphores prevent busy waiting by *suspending* a process that is waiting for a lock
	- It places the process into the waiting queue
	- `sleep()` suspends the process, `wakeup()` resumes the operation

- Each semaphore has an int value and list of processes
	- When a process needs to wait on a semaphore, it is added to the list of processes


