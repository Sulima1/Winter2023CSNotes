#CP386
## Storage Structure 
- Main memory - only large storage media that the CPU can access directly
	- Random Access
		- Typically, volatile
		- Typically, random-access memory in the form of Dynamic Random-access Memory (DRAM)
- Secondary Storage - extension of main memory that provides large nonvolatile storage capacity

**DMA** - Direct Memory Access
	eg: occurs when a USB requests access to a computer's harddrive through the CPU. When the USB is permitted access after sending an interrupt, it is given permission to access the harddrive aka DMA

![[Pasted image 20230126091704.png]]

![[Pasted image 20230126091728.png]]

## Direct Memory Access Structure 
- Used for high-speed I/O devices able to transmit infrmation at close to memory speeds
- Device controller transfers blocks of data from buffer storage directly to main memory without CPU intervention
- Only one interrupt is generated per block, rather than the one interrupt per byte
	- A block is generally 128MB, or any size agreed on when the CPU and the device first communicate

**Symmetric Multiprocessing Architecture** - two seperate CPUs sharing the same motherboard, each has their own registers and cache

![[Pasted image 20230126091808.png]]

## Dual Core Design
- Multi-chip and multi-core 
- Systems containing all chips
	- Chassis containing multiple seperate systems

## Non-Uniform Memory Access System
- The extent of sharing of storage can create different multi-processor systems


## Clustered Systems
- Like multiprocessor systems, but multiple systems working together
	- Usually sharing storage via a storage-area network (SAN)
	- Provides a high availability service which survives failures
		- Asymmetric clustering has one machine in hot-standby mode
		- Symmetric clustering has mutliple nodes running applications. monitoring each other
	- Some clusters are high performance computer (HPC)
		- Applications must be written to use parallelization
	- Some have distributed lack manager (DLM) to avoid conflicting operations


## Multiprogramming (Batch system)
- Single user cannot always keep CPU and I/O devices busy
- Multiprogramming organizes jobs (code and data) so CPU always has one to execute
- A subset of total jobs in system is kept in memory
- One job selected and run via job scheduling
- When job has to waite (for I/O fr example), OS switches to another job

![[Pasted image 20230126091916.png]]

## Multitasking (Timesharing)
- A logical extension of Batch systems - the CPU switches jobs so frequently that users can interact with each job while it is running, creating interactive computing
	- Response time should be < 1 second
	- Each user has at least one program executing in memory
	- If several jobs ready to run at te same time, use CPU scheduling
	- If processes don't fit in memory, swapping moves them in and out to run
	- Virtual memory allows execution of processes not completely in memory


## Dual-mode Operation
- Dual-mode (hardware supported) opeartion allows OS to protect itself and other system components
	- User mode and kernel mode
- Mode bit provided by hardware
	- Provides ability to distinguish when system is running user code or kernel code
	- When a user is running, mode bit is "user" (1)
	- When kernel code is executing, mode bit is "kernel" (0)

 - Typically, applications run in user mode and core operating system components run in kernel mode (also known as supervisor mode) where the OS executes its own code 

- How do we guarantee that user does not explicitly set the mode bit to "kernel"?
	- System call changes mode to kernel, return from call resets it to user
- Some instructions designated as privileged and protects the operating system from errant users  - and errant users from one another. The hardware allows privileged instructions to be executed only in kernel mode

User process executing -> calls system call -> trap mode bit = 0 -> executes system call in kernel -> return mode bit = 1 -> returns from system call back in user mode

![[Pasted image 20230126091933.png]]