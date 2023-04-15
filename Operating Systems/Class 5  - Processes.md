- directories have 
**drwxr-xr-x** - d for directory, r for read, w for write, x for execute
- Files come in the form of **-rw-r--r--** 

- The count of Ds, Rs, and Ws sum up to tell the user what level of commands they have
- In posix, for RWX (read, write, execute) 
	- R = 4
	- W = 2
	- X = 1

- When a code throws and error number it helps us diagnose the problem and shows us who can access and execute the files
- For example, in the error code 750
	- 7 represents Owner
		- (R = 4) + (W = 2) + (X = 1)
	- 5 represents Group
		- (R = 4) + (X = 1)
	- 0 represents Others
		- 0 + 0 + 0


- That is why in assignment 1 0777 for 
```c
int createDir(char* dirName){

    int makeDir = mkdir(dirName, 0777);

    if (makeDir == 0){

        printf("Directory is created successfully.");

        return 0;

    }

    else{

        printf("mkdir error");

    }

    return 1;

}
```
Represents that we are creating a directory with 777 permissions, anyone can access it


- fseek reads the size of a file
```c
fseek(fp, 0L, SEEK_END);
```


## Process Concept

- An operating system has a variety of programs
	- Batch system - **jobs**
	- Time-shared systems - **users programs** or **tasks**
- **Process** - a program in execution
- Multiple Parts
	- The program code also called the **text section**
	- Current activity including **program counter**
	- **Stack** containing temporary data
	- **Data section** containing global variables
	- **Head** containing memory dynamically allocated during run time

| stack |
|          | - data transfer down
|          |  - data transfer up
| heap | 
| data  |
| text   |


## Process Control Block (PCB)
- also called the *task control block*
- Process state - running, waiting, etc
- Program counter - location of instruction to next execute
- CPU Registers - contents of all process centric registers

## Process Scheduling
- Process schedulers selects among available processes to use

- AN I/O bound process is one that spends more of its time doing I/O than it spends doing computations
	- Many short CPU bursts
- A CPU-Bound process, in contrast, generates I/O requests infrequently, using more of its time doing computations
	- Few very long CPU bursts
- Maintains scheduling queues
	- Job queue - set of all processes in the system
	- Ready queue - set of all processes residing in main memory
	- Device queue - set of prcesses watiing for an I/O device