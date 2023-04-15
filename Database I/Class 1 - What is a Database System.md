#CP363
## Database System

- Computerized record keeping system
- Supports operations
	- Insert, retrieve, remove, or change data in database
- Components
	- Data, hardware, software, users

## Data

- Data consists of raw facts
	- Not yet processed to reveal meaning to the end user
	- Building blocks of information
- Types of databases
	- Single-user database: supports one user at a time
	- Multi-user database: supports mutliple users at a time

- Database system may support single user (for small machines) or many users
- When there are many users in organizations:
	- Data is integrated: database is unification of distinct files. Any redundency among the files partly or wholly iseliminated
	- Data is sahred: Different users have access to the same data
- Different users require different views

- For example a given database can have EMPLOYEE file that shows the information of employees. Also this database can contain an ENROLLMENT file that shows the enrollment of eomployees in training courses
- Personnel department uses EMPLOYEE and education department uses ENROLLMENT files

EMPLOYEE
| Name | Address | Department | Salary |
|-------|-----------|---------------|--------|

ENROLLMENT
| Name | Course | ...|
|--------|-|-|


## Hardware 

- The hardware components of database systems consist of the disks in which data are stored thus the database system can perform
	- Direct access to subset portions of data
	- Rapid I/O
- Data operated on in the main memory

## Software

- Between physically stored data and users of the systems there is a layer of stware referred to as 
1. Database manager
2. Database server
3. Database management system (DBMS)
- DBMS shields database users from hardware details

## Users
- DBA - Database administrator 

## What is a database
- Collection of persistent data that is used by the application systems of some given enterprise 

