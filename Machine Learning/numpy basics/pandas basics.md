#CP322 #Python 

### Dataframes
- A table in which each column has a type
```python
animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]})
```


| | Cows | Goats |
| - |- | -| 
| 0 | 12 | 22
|1 | 20 | 19 


- Can also be formatted like this

```python
fruits = pd.DataFrame([(30, 21), (20,24), (34,12)], columns=['Apples', 'Banana'])
fruits
```

| | Apples | Bananas |
|-|-|-|
|0|30|21|
|1|20|24|
|2|34|12|


- Indexes can be named 
```python
fruit_sales = pd.DataFrame([(35,21), (20,24), (34,12)], columns=['Apples', 'Bananas'], index=["2017 Sales", "2018 Sales", "2019 Sales"])
```

 - You can import CSV files and datasets from github for example and format them using
 ```python
URL="https://raw.githubusercontent.com/sukhjitsehra/datasets/master/DATA200/Datasets/Beers.csv"
beers= pd.read_csv(URL, index_col = False)
beers.head() # To look at first few rows of records of dataset
```

