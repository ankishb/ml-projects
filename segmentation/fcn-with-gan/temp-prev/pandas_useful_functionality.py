








>>> df = pd.DataFrame([('bird',    389.0),
...                    ('bird',     24.0),
...                    ('mammal',   80.5),
...                    ('mammal', np.nan)],
...                   index=['falcon', 'parrot', 'lion', 'monkey'],
...                   columns=('class', 'max_speed'))
>>> df
         class  max_speed
falcon    bird      389.0
parrot    bird       24.0
lion    mammal       80.5
monkey  mammal        NaN

When we reset the index, the old index is added as a column, and a new sequential index is used:

>>> df.reset_index()
    index   class  max_speed
0  falcon    bird      389.0
1  parrot    bird       24.0
2    lion  mammal       80.5
3  monkey  mammal        NaN











join the two dataframes along columns

pd.concat([df_a, df_b], axis=1)

    subject_id  first_name  last_name   subject_id  first_name  last_name
0       1         Alex      Anderson        4         Billy       Bonder
1       2         Amy       Ackerman        5         Brian       Black
2       3         Allen     Ali             6         Bran        Balwner
3       4         Alice     Aoni            7         Bryce       Brice
4       5         Ayoung    Atiches         8         Betty       Btisan


Merge two dataframes along the subject_id value

pd.merge(df_new, df_n, on='subject_id')

   sub_id  f_name  last_name   test_id
0   1       Alex    Anderson            51
1   2       Amy     Ackerman            15
2   3       Allen   Ali                 15
3   4       Alice   Aoni                61
4   4       Billy   Bonder              61
5   5       Ayoung  Atiches            16
6   5       Brian   Black               16
7   7       Bryce   Brice               14
8   8       Betty   Btisan              15











You can get a rather good score after creating some lag-based features like in advice from previous week and feeding them into gradient boosted trees model.

Apart from item/shop pair lags you can try adding lagged values of total shop or total item sales (which are essentially mean-encodings). 
All of that is going to add some new information.

# Lagged values by one row
df['previous_days_stock_price'] = df['stock_price'].shift(1)











>>> df
   A  B      C
0  1  1  0.362838
1  1  2  0.227877
2  2  3  1.267767
3  2  4 -0.562860





>>> df.groupby('A').agg(['min', 'max'])
    B             C
  min max       min       max
A
1   1   2  0.227877  0.362838
2   3   4 -0.562860  1.267767






Different aggregations per column

>>> df.groupby('A').agg({'B': ['min', 'max'], 'C': 'sum'})
    B             C
  min max       sum
A
1   1   2  0.590716
2   3   4  0.704907




inplace=True means that the changes are saved to the df right away
