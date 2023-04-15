#CP363 

## Steps for Design of Conceptual Schema
1. Choice of model: User requirements and real world concepts should go into design model
2. Normalization: adjust diagrams and normalize relational model
3. Optimization: based on data dictionary and database description

## Entity Relationship Model (ER)
- Introduced by Peter Chen in 1976
- Consists of:
	- Entities
	- Relationships
	- Attributes
- No single, generally accepted notation

## Entities
- A representation of a real world object that holds data or information
- There are two types of entities
	- Strong/regular identity
	- Weak entity
- Strong entity types are called **owner** or **dominant**
- Weak entity types are called **dependent** or **subordinate**
	- Existance of a weak entity depends on the existence of a strong identity

- Weak entity types are represented by a double outlined box, single outlined box represents a strong entity


## Relationships
- A relationship associates two entities together
- Mapping constraints
	- one-to-one 
	- one-to-many
	- many-to-many


## Attributes/Properties
- Describes the properties of entities and relationships
- Types of attributes
	- Simple or composite
	- Single-value or multi-value
	- Stored/based or derived
- An attribute can be a key or non-key
- Attributes can also have null values in some cases