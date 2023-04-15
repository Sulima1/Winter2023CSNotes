#CP386
## What is an operating system
- A program that acts as an intermediary between a user of a computer and the computer hardware
	- Execute user programs and make solving user problems easier
	- Make the computer system convenient to use
	- Use the computer hardware in an efficient manner

## CPU 
- **CPU** is the manager for the computer that is controlled by the OS
	- Manager because it allocates the tasks to each component equally, similar to a manager


## Computer System Structure
- Computer system can be divided into four components
	- Hardware - controls basic functions
		- CPU, memory, I/O devices
	- Operating System
		- Controls various coordinates use of hardware among various applications and users
	- Application programs - define the ways in which the system resources are used to solve the computing problems of users
		- Word processors, compilers, web browsers, database systems, video games
	- Users
		- People, machines, other computers

## What Operating Systems do
- Users of dedicate systems such as workstations have dedicated resources but frequently use shared resources from servers  
 - Mobile devices like smartphones and tablets are resource poor, optimized for usability and battery life  
	 - Mobile user interfaces such as touch screens, voice recognition  

- Some computers have little or no user interface, such as embedded computers in devices and automobiles  
	- Run primarily without user intervention


## Defining Operating Systems
- Term OS covers many roles
	- Present in toasters through ships, spacecraft, game machines, TVs and industrial control systems
	- Born when fixed use computers for military became more general purpose and needed resource management and program control


## Operating System Definition
- Everything is either
	- A system program (ships with the operating system, but not part of the kernel)
	- An application program, all programs not associated with the operating system


# Overview of Computer System Structure

## Computer System Organization
- One of more CPUs, device controllers connect through common bus providing access to shared memory 
- Concurrent execuation of CPUs and devices competing for memory cycles
![[Pasted image 20230126091533.png]]


## Computer System Operation
- Each device controller is in charge of a particular device type
- Each device controller has a local buffer
- Each device controller type has an operating system **device driver** to manage it
- CPU moves data from/to main memory to/from local buffers

![[Pasted image 20230126091603.png]]
## Common Functions of Interrupts
- Interrupt transfers control to the interrupt service routine generally, through the interrupt vector, which contains the addresses of all the service routines
- Interrupt architecutre must save the address of the interrupted construction
- A trap or exception is a software-generated interrupt caused either by an error or a user request
- An operating system is interrupt driven


## Interrupt Handling
- The CPU hardware has a wire called the interrupt-request line that the CPU sense after executing every instruction
- When the CPU detects that a controller has asserted a signal on the interrupt-request line
	- The operating system preserves the state of the CPU by storing the registers and the program counter
	- It reads the interrupt number and jumps to the interrupt-handler routine by using that interrupt number as an index into the interrupt vector
	- It then starts execution at the address associated with that index
	- It performs a state restore and executes a return from interrupt instruction to return the CPU to the execution state prior to the interrupt

## Interrupt-drive I/O Cycle
- Most CPUs have two interrupt request lines
	- One is the non-maskable interrupt, which is reserved for events such as unrecoverable memory errors
	- The second interrupt line is maskable: it can be turned off by the CPU before the execution of critical instruction sequences that must not be interrupted
		- The maskable interrupt is used by device controllers to request service
	- The interrupt mechanism also implements a system of interrupt priority levels
		- These levels enable the CPU to defer the hanlding of low-priority interrupts without masking all interrupts and makes it possible for a high-priority interrupt to preempt the execution of a low-priority interrupt
![[Pasted image 20230126091620.png]]

## Checking Interrupts on Linux OS
- The file /proc/interrupts contains information about the hardware interrupts in use and how many times the processor has been interrupted
- watch -n1 "cat /proc/interrupts"
- The first column indicated the IRQ (Interrupt Request Line) Number. Subsequent columns indicate how many interrupts have been generated for the IRQ number on different CPU cores. The last column provides information on the programmable interrupt controller that handles the interrupt
- A small IRQ number value means higher priority

- The software interrupts are generated wehn the CPU executes an intruction which can cause exception condition in the CPU [ALU Unit] itself
- watch -n1 "cat /proc/stat"
- The "intr" line gives counts of interrupts services since boot time, for each possible system interrupts 