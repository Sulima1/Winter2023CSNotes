(min, max) relational notation
(0,1) min of 0 employees managed by 1 employee
(1,1) max of 1 employee managing 1 employee
(1,N) max of 1 employee managing many

## Alternative Notation for Structural Constraints
- Associate a pair of integer numbers
	- (min, max) where 0 <= min <= max
	- Each entity must participate in at least min and at most max relationship instances at all times

## Subclass and Superclass
- Subclass(Subtype) and superclass(Supertype)
	- Student entity type can have part time and full time student subclasses
	- Student in this case is the superclass
	- Attribute inheritance
		- Member of a subclass inherits all the attribute of its suplerclass 
		- Each subclass can have its own attributes 
			- In addition to the inherited attributes

## Specialization
- Process of defining a set of subclasses of an entity type
	- Usually based on some distinguishing characteristic of entity type
	- **account** can be specialized into **savings-account** and **chequeing-account**


## Disjoint and Overlapping
- In a disjoint specialization, also called an exclusive specialization, an individual of the parent class may be a member of only one specialized subclass
- In an overlapping specialization, an individual of of the parent class may be a member of more than one of the specialized subclasses
- A part can be manufactured AND purchased (overlapping) but an individual can ONLY be a technician or a secretary (disjoint)

## Generalization
- Result of taking the union of two or more lower level entities to produce a higher level entity
	- The original entities are subclasses and the combined result is a superclass

- In the ER model, a double line represents that the two sub entities that were combined are a part of the whole
	- Cars and trucks were combined to create the vehicle super class
	- All cars and trucks are vehicles
