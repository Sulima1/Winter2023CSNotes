#CP386
## Operating System Services
![[Pasted image 20230126092118.png]]
- An environment for execution of programs and services to program and users
- Providing functions that are helpful to the user
	- User interface
	- Program execution
	- I/O Operations
	- File-system manipulation
	- Communications
	- Error detection

- Another set of OS functions exists for ensuring the efficient operation of the system itself via resource sharing
	- Resource allocations - **CPU**
	- Protection and security
		- Protection: involves ensuring that all access to system resources is controlled
		- Security: of the system from outsiders requires user authentication, extends to defending external I/O devices from invalid access attempts

- User Interfaces provided by the OS
	1. GUI
	2. batch
	3. command line

- System calls are an API which has permissions to invoke kernel based services

## CLI
- CLI or **command interpreter** allows direct command entry
	- Sometimes multiple flavors implemented - **shells**
	- Primarily fetches a command from user and executes it
	- Sometimes commands built-in. sometimes just names of programs

## System Calls
- Implemented by a **system call interface**
- Programming interface to the services provided by the OS
- Typically written in a high-level language (C or C++)
- Mostly accessed by programs via a high-level Applications Programming Interface (API) rather than direct system call use
- Three most common APIs are Win32 API for Windows, POSIX API for POSIX-based systems (including virtually all versions of UNIX, Linux and Mac OS X), and Java API for the Java virtual machine (JVM)

## API
- A programmer accesses an API via library of code provided by the operating system. in the case of UNIX and Linux for programs written in the C language we use **libc**
- APIs provide portability and are easy to work with
![[Pasted image 20230126092142.png]]