#CP386

- Virtual memory is a technique that allows for the execution of processes that are not completely in memory

## 10.1
- **virtual memory** is the seperation of logical memory and physical memory
- The **virtual address space** of a process refers to the logical (or virtual) view of how a process is stored in memory

- Virtual address spaces that include holes are known as **sparse** address spaces

## 10.2
- A strategy to load pages is to use **demand paging** where pages are loaded only when they are *demanded* during program execution

### Basic Concepts
- Access to a page marked invalid causes a **page fault**
- The procedure to handle a page fault is:
	- Check the internal table to determine whether the reference was valid or invalid
	- If it was invalid, terminate the process. If it was validm bring the page in
	- Find a free frame
	- Schedule a secondary storage operation to read the desired page into the newly allocated frame
	- When the storage read is complete, modify the internal table kept with the process and the page table to indicate the page is now in memory
	- Restart the instruction that was interrupted by the trap. The process can now access the page as normal

- **pure demand paging** is when you never bring a page into memory until it is required
- Programs tend to have a **locality of reference**, which results in reasonable performance from demand paging
- The hardware to support demand paging is the same as hardware for paging and swapping
	- **Page table**: the table has the ability to mark an entry invalid through a valid-invalid or special value of protection bits
	- **Secondary memory** The memory holds those pages that are not present in main memory. The secondary memory is usually a high speed disk or NVM device