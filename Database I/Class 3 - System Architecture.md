#CP363 

## Schema

- Database Schema
	- Describes the database
	- Specified during the design phase
	- Most data models have a graphical representation


### Example: SUPPLIER-PARTS DB
SUPPLIER
| Supplier # | Supplier_name | City|
|---|--|--|

PART 
| Part # | Part_name | Weight |
|-|-|-|

PARTS_SUPPLIER
| Part # | Supplier # | Quantity|
|-|-|-|


## Instances
- Refers to the database at a particular moment in time
- Every time we insert, delete, or update the value of a data item, we change on instance of database to another
- DBMS is responsible for ensuring that every instance satisfies
	- Structure and constraints satisfied in schema


## Three levels of architecture
- Also known as the ANSI/SPARC architecture or three schema architecture
- Framework used for describing the structure of specific database systems

### The Three levels
1. Internal level: Shows how data are stored inside the system
	1. It is the closest level to physical storage
	2. The relational model has *nothing* explicit to say about the internal level
2. Conceptual level: Deals with the modeling of the whole database
3. External level: Models a user oriented description part of the database


## Mapping
- The key for describing data independence
	- **Data Independence**: is the capacity to change the schema at one level without having to change the schema at a higher level
- The two types of data independence are
	- Logical data independence
	- Physical data independence


### Logical Data Independence
- Ability to modify conceptual schema without changing
	- External views
	- Application programs
- Changes to conceptual schema may be necessary whenever the logical structure of the database changes

### Physical Data Independence
- Ability to modify internal or physical schema without changing
	- Conceptual or view level schema
	- Application programs
- Changes to physical schema may be necessary to improve performance of retrieval or update
- Achieving logical data independence is more difficult than physical data independence


## Database Administrator
- Participates in conceptual databse design
- Determines how to implement conceptual schema
- Teach users, and helps them report
- Implements security and integrity 
- Implements unload/reload utilities
- Monitor and tune database performance


## DBMS Languages
- Users interact with database with data sublanguage
	- DDL: To define the database
		- Required in building databases
		- Commonly referred to as *data definition language*
	- DML: To manipulse data
		- Used to construct and use the database
		- Referred to as *data manipulation language


## Database Management System
- DDL processor / compiler
- DML processor / compiler
- Handle scheduled and *ad hoc* inquiries
- Optimizer and run-time manager
- Security and integrity
- Recovery and concurrency
- **Data dictionary** is a system database that contains "data about the data" 
	- Definitions of other objects in the system, also known as metadata

## Support for System Processes
- Data Communications interface
- Client Server Architecture
- External tool support: query, reports, graphics, spreadsheets, statistics
- Utilities: unload/reload, stats, re-org
- Distributed processing



