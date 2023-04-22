#CP386 

## 7.1

### The Bounded-Buffer Problem
- The bounded buffer problem happens when one process produces data and places it into a buffer while another process consumes data from the buffer
- The challenge is to prevent the buffer from being written to when it is full, and read from when it is empty
- Solution to the bounded buffer problem using semaphores:
```c
int n;
semaphore mutex = 1;
semaphore empty = n;
semaphore full = 0;
```

### The Readers-Writers Problem
- The readers-writers problem happens when there are multiple read and write processes who want access to a shared resource simultaneously
- requires that multiple readers can access a shared resource without conflict but a writer requires exlcusive access to the resource

- It has a couple of variations
	- The *first* readers-writers problem requires that no reader be kept waiting unless a writer has already obtained permission to use the resource
	- The *second* readers-writers problem requires that once a writer is ready, the writer performs its task as soon as possible
		- If a writer is waiting to access a resource, no new readers can start reading
- In the first case, writers starve and in the second case the readers starve

- Solution to the readers-writers problem
```c
semaphore rw_mutex = 1;
semaphore mutex = 1;
int read_count = 0;

while (true){
	wait(rw_mutex);
	/* writing is being performed */
	signal(rw_mutex)
}
```

- Reader-writer locks are useful when:
	- used in applications where it is easy to identify which processes only read and which only write
	- used in applications that have more readers than writers


### The Dining-Philosophers Problem
- 5 philosophers sit at a round table, each with one chopstick on the left and right side (5 philosophers, 5 individual chopsticks)
- If all philosophers grab their left chopstick they will suceed, but if they try and then grab the right chopstick none of them will be able to 
	- This is a **deadlock situation**


##### The Semaphore Solution
- One solution is to represent each chopstick with a semaphore `semaphore chopstick[5];`
	- This will inevitably lead to a deadlock
- If all the philosophers grab their left chopstcik, the value of the semaphore will now be 0 causing any attempts to grab another chopstick to be delayed forever

- Remedies to the deadlock solution:
	- Allow at most four philosophers to be sitting simultaneously at the table
	- Allow a philosopher to pick up her chopsticks only if both chopsticks are available (to do this, they must be picked up in a critical section)
	- Use an asymmetric solution - an odd numbered philosopher picks up first her left chopstick then her right, whereas an even numbered picks up her right and then her left


## 7.2 
- For thread synchronization outside the kernel, windows provides **dispatcher objects**
	- Allows threads to synchronize to different mechanisms such as mutex locks, semaphores, events, timers etc
- Events are similar to condition variables; they may notify a waiting thread when a desired condition occurs
- Timers are used to notify one thread that a specific amount of time has passed

- Dispatcher objects can be in a **signaled state** when it is available and a thread will not be blocked when acquiring an object
- If objects are in a **nonsignaled state**, it is not available and threads will be blocked from acquiring them
![[Pasted image 20230417170550.png | 350]]

- The kernel selects only one thread from the waiting queue for a mutex, because a mutex can only be "owned" by a single thread6

- A **critical section object** is a user-mode mutex that can be acquired and released without kernel intervention

- In the Linux kernel, both spinlocks and mutex locks are **nonrecursive** meaning that if a thread has acquired one of these locks, it cannot acquire the same lock a second time without releasing the first lock
- Linux supplies `preempt_disable()` and `preempt_enable()` to disable and enable kernel preemption


## 7.3
### Posix Mutex
- A mutex is a lock that protects critical sections of code
- `pthread_mutex_init()` initializes a mutex lock

```c
#include <pthread.h>

pthread_mutex_t mutex;

pthread_mutex_init(&mutex, NULL)
```

- Mutexes are acquired with `pthread_mutex_lock()` and released with `pthread_mutex_unlock()`


### Posix Semaphore
- Posix specifies two types of semaphores: **named** and **unnamed**


#### Created a named semaphore
```c
#include <semaphore.h>
sem_t *sem;

sem = sem_open("SEM", O_CREAT, 0666, 1);
```
- The advantage of named semaphores is that multiple unrelated processes can easily use the semaphore just by calling its name


#### Created an unnamed semaphore
- Unnamed semaphores are passed three parameters
	- A pointer to the semaphore
	- A flag indicating the level of sharing
	- The semaphore's initial value

```c
#include <semaphore.h>
sem_t sem;

sem_init(&sem, 0, 1)
```

- In this case, the flag of the semaphore is set to 0 meaning that only threads belonging to the process that created the semaphore can use it
	- If we were to set the flag to 1, it would be placed in shared memory

- Both named and unnamed semaphores use `sem_wait()` and `sem_post()`


#### Posix Condition Variables