#CP386 

## System Call Parameter Passing
- Three general methods used to pass parameters to the OS
	- Simplest: Pass the parameters in registerst
	- Parameters stored in a block or table in memory and address is passed to register
	- Parameters places, or pushed onto the stack by the program and popped off the the stack by OS

![[Pasted image 20230126092327.png]]

## Types of System Calls

### Information Maintenance
- get time or date, set time or date
- get system data, set system data
- get and set process or device attributes

### Communications
- create, delete communicaition connection
- send, receive messages if **message passing model** to **host name** or **server name**
	- from **client** to **server** (client server architecture)
- **shared memory model** to create and gain access to memory regions
-  transfer status information


**embedded systems** - are systems that are built around a microcontroller instead of a microprocessor


## Link and Loaders
- Steps in the process of converting written C code to a runnable object file
- A **loaders** is used to load the binary executable file into memory where it is eligible to run on a CPU core
- An activity associated with linking and loading is relocation, which assigns final addresses to the program parts and adjusts code and data in the progra to match those addresses


## Linux System Structure
- Despite the apparent simplicity of monolithic kernels (such as UNIX), they are difficult to implement and extend

## Layered Approach
- Loosely coupled and the changes in one component affect only that component, allowing system implementers more freedom in creating changes in operating system

## Microkernel System Architecture
- Instead of keeping everything in the kernel space, why not move some of the kernel space contents into the user space
	- Makes the kernel lighter
	- **Mach** example of microkernel
- Communicatoin takes place between user modules using **message passing**

- Easier to extend microkernel
- Easier to port the OS to new architectures
- More reliable (less code is running in kernel mode)


## Hybrid Systems
- Hybrid combines multple approaches to address performance, security, usability needs
- Linux and Solaris kernels in kernel address space, so monolithic, plus modular for dynamic loading of functionality


## System Boot
- Bootloader is the first program that executes after power, and it never relies on the kernel
- Bootstrap is the second stage after bootloader, belongs to kernel code that act as a link between bootloader and kernel mirroring

- bootloader -> bootstrap (HEAD.O) -> kernel vmlinux (HEAD.O) -> kernel start_kernel (MAIN.O)

