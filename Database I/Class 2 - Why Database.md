#CP363
- **Shared Data** - not only for existing applications but also new ones canbe deeveloped to operated agains the same data

- **Reduced Redundancy** - if we need to keep some redundancies it should be controlled

- **Reduced Inconsistent Data**: Inconsistency happens when one of the redundant data has been updated and the other has not. Propagating updates is used to avoid inconsistency

- **Transaction support**: By putting several update operations in one transaction we can guarantee that either all of them are done or none of them are done. For example are done or none of them are done. For example the operations fro transferring cache from account A to account B can be put in one transaction


- Support for data integrity: Ensures that the data in database is accurate
- For example:
	- We shouldn't have an employee working in non exiting department
	- We shouldn't have number of hours entered as 400 instead of 40
	- Inconsistency can lead to the lack of integrity

- Security Enforcement: Ensuring that the only means of access to a database is through proper channels by:
	- Restricting unauthorized data
	- Different checks (security constraints) can be establishe for each type of access (retreive, insert, delete, etc)
	- Example: Course marks database
		- A student can have access to his/her own mark
			- Should not be able to see other student's marks
		- TA might have access to enter marks for the current assignment only
			- Should not be allowed to change marks for the other assignments/tests
		- Instructor can have full course to the database 

- Support for standards
	- Due to central control of database
	- eg: an address must be two lines, each 40 characters maximum

- Conflicting requirements can be met
	- Knowledge of overall enterprise requirements as opposed to individual requirements
		- System can be designed to provide overall service that is best for the enterprise
		- Data representation can be selected that is good for most important applications (at the cost of some other applications)

## Data independance

- Data independance
	- Traditional file processing is data-dependent
		- Knowledge of data organization and access technique is built into application logic and code
- Examples of situations in which the stored representations might be subject to change
	- An application program written to search a student file in which records are sorted in ascending order by student# fails if the sort order is reversed
	- Representation of numeric data:
		- Binary or decimal or BCD
		- Fixed or floating point
		- Real or complex

- Representation of characters
	- ASCII (uses 1 byte)
	- Unicode (uses 2 bytes)
	- Universal character set (USC)
		- UCS 2 (uses 2 bytes)
		- UCS 4 (uses 4 bytes)
	- Unit for numeric data
		- Inches or centimeters 
		- Pounds or kilograms
	- Data encoding
		- Red = 1, Blue = 2, ....

- In database systems DBMS immune applications to such changes
- In database systems the logical and physical representation of data are seperated
- Using database allows changes to application programs without changing the structure of the underlying data and vice versa
	- The database can grow without impairing existing applications
	- For example: without requiring any changes to the existing applications a "unit cost" *field* can be added to the "part" *stored record* in the "parts"*stored file* of the database shown in Fig 1.7 of the text book


## Relational Systems
- Introduction of relational model in 1969-70 was the most importatnt innovation in database history
- Relational systems are based on logic and mathematics. Relation is basically a mathematical term for a table
- In relational systems data is perceived as tables only and operators derive new tables from existing
- Relational systems are not pointer based (to the user) although they may use pointers at the level of physical implementation


## Relational Products
- They appeared in the market in the late 70s and mostly support SQL.
- The names of some of these products which are based on the relational system are:
	• DB2 from IBM Corp.  
	• Ingres II from Computer Associate International Inc.  
	• Informix from Informix Software Inc.  
	• Microsoft SQL Server from Microsoft Corp.  
	• Oracle 11g from Oracle Corp.  
	• Sybase Adaptive Server from Sybase Corp.


## Non Relational Systems
• Hierarchic  
• Network  
• Inverted List  
• Object  
• Object/Relational  
• Multi-dimensional

## DBMS Functions
- The functions of DBMS may be summarized as follows:
	- Data dictionary management: stores the definitions of data and their relationships (metadata) in data dictionary; any changes made are automatically recorded in the data dictionary
	- Security Management: Enforces user security and data privacy
	- Multiuser access control: Sophisticated algorithms ensure that multiple users can access the database concurrently without compromising its integrity


## Disadvantages of database systems
- Increasing cost: Database systems require sophisticated hardware and software and highly skilled personnel
- Management complexity: Database systems interface with many different technologies
- Vendor dependence: Given the heavy investment in technology and personnel training, companies might be relucatnt to change database vendors
- Frequent upgrade/replacement cycles: DBMS vendors frequently upgrade their products by adding new functionality. Some of these versions require software and/or hardware upgrades which cost money