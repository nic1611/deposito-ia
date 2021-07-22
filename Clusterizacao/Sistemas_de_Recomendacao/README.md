![alt text](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTK4gQ9nhwHHaSXMHpeggWg7twwMCgb877smkRmtkmDeDoGF9Z6&usqp=CAU)
# <font color='Blue'> Ciência dos Dados na Prática</font>



# Sistemas de Recomendação

![](https://img.icons8.com/emoji/452/books-emoji.png)

Cada empresa de consumo de Internet precisa um sistema de recomendação como **Netflix**, **Youtube**, **feed de notícias**, **Site de Viagens e passagens Aéreas**, **Hotéis**, **Mercado livre**, **Magalu**, **Olist**, etc. O que você deseja mostrar de uma grande variedade de itens é um sistema de recomendação.

## O que realmente é o Sistema de Recomendação?

Um mecanismo de recomendação é uma classe de aprendizado de máquina que oferece sugestões relevantes ao cliente. Antes do sistema de recomendação, a grande tendência para comprar era aceitar sugestões de amigos. Mas agora o Google sabe quais notícias você vai ler, o Youtube sabe que tipo de vídeos você vai assistir com base em seu histórico de pesquisa, histórico de exibição ou histórico de compra.

Um sistema de recomendação ajuda uma organização a criar clientes fiéis e construir a confiança deles nos produtos e serviços desejados para os quais vieram em seu site. Os sistemas de recomendação de hoje são tão poderosos que também podem lidar com o novo cliente que visitou o site pela primeira vez. Eles recomendam os produtos que estão em alta ou com alta classificação e também podem recomendar os produtos que trazem o máximo de lucro para a empresa.

Um sistema de recomendação de livros é um tipo de sistema de recomendação em que temos que recomendar livros semelhantes ao leitor com base em seu interesse. O sistema de recomendação de livros é usado por sites online que fornecem e-books como google play books, open library, good Read's, etc.

# 1° Problema de Negócio

Usaremos o método de **filtragem baseada em colaboração** para construir um sistema de recomendação de livros. Ou seja, precisamos construir uma máquina preditiva que, **com base nas escolhas de leituras de outras pessoas, o livro seja recomendado a outras pessoas com interesses semelhantes.**



Ex:

**Eduardo** leu e gostou dos livros A loja de Tudo e Elon Musk.

**Clarice** também leu e gostou desses dois livros



![](https://d1pkzhm5uq4mnt.cloudfront.net/imagens/capas/c3d5f7a1b2ed4261d4bdfdabe795cdf5bb8bf58d.jpg)![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTpahWGbsRS5_E16Ad8nrh5897oVznxv-hX6SLyJlyOvjDpQnlDUQZTPGqfS_a7qX7Z7gg&usqp=CAU)



Agora o **Eduardo** leu e gostou do livro "StartUp de U$100" que não é lido pela **Clarice**. 


![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSzauvLe3kPPG_TkgiPhy51rAXzXbj46SKDSMbchCyKB12pTcNZBSn8KfxIgrgHA-YN56k&usqp=CAU)


Então **temos que recomendar o livro **"StartUp de U$100" para **Clarice**




## **Resultado**

   Você concorda que se vc receber uma recomendação certeira, a chance de vc comprar o livro é muito maior?

   Vc concorda que se mais pessoas comprarem, maior será o faturamento da empresa?

   Vc concorda que os clientes vão ficar muito mais satisfeitos se o site demonstrar que conhece ela e que realmente só oferece produtos que realmente são relevantes p ela?

# 2° Análise Exploratória dos Dados



```python
#Importação das Bibliotecas ou Pacotes
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
```

Fonte de Dados:

https://www.kaggle.com/rxsraghavagrawal/book-recommender-system

#### Base de Livros


```python
# Importação dos Dados Referentes aos Livros
books = pd.read_csv("BX-Books.csv", sep=';', encoding="latin-1", error_bad_lines= False)

```
    


```python
books
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ISBN</th>
      <th>Book-Title</th>
      <th>Book-Author</th>
      <th>Year-Of-Publication</th>
      <th>Publisher</th>
      <th>Image-URL-S</th>
      <th>Image-URL-M</th>
      <th>Image-URL-L</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0195153448</td>
      <td>Classical Mythology</td>
      <td>Mark P. O. Morford</td>
      <td>2002</td>
      <td>Oxford University Press</td>
      <td>http://images.amazon.com/images/P/0195153448.0...</td>
      <td>http://images.amazon.com/images/P/0195153448.0...</td>
      <td>http://images.amazon.com/images/P/0195153448.0...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0002005018</td>
      <td>Clara Callan</td>
      <td>Richard Bruce Wright</td>
      <td>2001</td>
      <td>HarperFlamingo Canada</td>
      <td>http://images.amazon.com/images/P/0002005018.0...</td>
      <td>http://images.amazon.com/images/P/0002005018.0...</td>
      <td>http://images.amazon.com/images/P/0002005018.0...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0060973129</td>
      <td>Decision in Normandy</td>
      <td>Carlo D'Este</td>
      <td>1991</td>
      <td>HarperPerennial</td>
      <td>http://images.amazon.com/images/P/0060973129.0...</td>
      <td>http://images.amazon.com/images/P/0060973129.0...</td>
      <td>http://images.amazon.com/images/P/0060973129.0...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0374157065</td>
      <td>Flu: The Story of the Great Influenza Pandemic...</td>
      <td>Gina Bari Kolata</td>
      <td>1999</td>
      <td>Farrar Straus Giroux</td>
      <td>http://images.amazon.com/images/P/0374157065.0...</td>
      <td>http://images.amazon.com/images/P/0374157065.0...</td>
      <td>http://images.amazon.com/images/P/0374157065.0...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0393045218</td>
      <td>The Mummies of Urumchi</td>
      <td>E. J. W. Barber</td>
      <td>1999</td>
      <td>W. W. Norton &amp;amp; Company</td>
      <td>http://images.amazon.com/images/P/0393045218.0...</td>
      <td>http://images.amazon.com/images/P/0393045218.0...</td>
      <td>http://images.amazon.com/images/P/0393045218.0...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>271355</th>
      <td>0440400988</td>
      <td>There's a Bat in Bunk Five</td>
      <td>Paula Danziger</td>
      <td>1988</td>
      <td>Random House Childrens Pub (Mm)</td>
      <td>http://images.amazon.com/images/P/0440400988.0...</td>
      <td>http://images.amazon.com/images/P/0440400988.0...</td>
      <td>http://images.amazon.com/images/P/0440400988.0...</td>
    </tr>
    <tr>
      <th>271356</th>
      <td>0525447644</td>
      <td>From One to One Hundred</td>
      <td>Teri Sloat</td>
      <td>1991</td>
      <td>Dutton Books</td>
      <td>http://images.amazon.com/images/P/0525447644.0...</td>
      <td>http://images.amazon.com/images/P/0525447644.0...</td>
      <td>http://images.amazon.com/images/P/0525447644.0...</td>
    </tr>
    <tr>
      <th>271357</th>
      <td>006008667X</td>
      <td>Lily Dale : The True Story of the Town that Ta...</td>
      <td>Christine Wicker</td>
      <td>2004</td>
      <td>HarperSanFrancisco</td>
      <td>http://images.amazon.com/images/P/006008667X.0...</td>
      <td>http://images.amazon.com/images/P/006008667X.0...</td>
      <td>http://images.amazon.com/images/P/006008667X.0...</td>
    </tr>
    <tr>
      <th>271358</th>
      <td>0192126040</td>
      <td>Republic (World's Classics)</td>
      <td>Plato</td>
      <td>1996</td>
      <td>Oxford University Press</td>
      <td>http://images.amazon.com/images/P/0192126040.0...</td>
      <td>http://images.amazon.com/images/P/0192126040.0...</td>
      <td>http://images.amazon.com/images/P/0192126040.0...</td>
    </tr>
    <tr>
      <th>271359</th>
      <td>0767409752</td>
      <td>A Guided Tour of Rene Descartes' Meditations o...</td>
      <td>Christopher  Biffle</td>
      <td>2000</td>
      <td>McGraw-Hill Humanities/Social Sciences/Languages</td>
      <td>http://images.amazon.com/images/P/0767409752.0...</td>
      <td>http://images.amazon.com/images/P/0767409752.0...</td>
      <td>http://images.amazon.com/images/P/0767409752.0...</td>
    </tr>
  </tbody>
</table>
<p>271360 rows × 8 columns</p>



#### Base de Usuários


```python
# Importação dos Dados Referentes aos Usuários
users = pd.read_csv("BX-Users.csv", sep=';', encoding="latin-1", error_bad_lines= False)

```
    


```python
users
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User-ID</th>
      <th>Location</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>nyc, new york, usa</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>stockton, california, usa</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>moscow, yukon territory, russia</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>porto, v.n.gaia, portugal</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>farnborough, hants, united kingdom</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>278853</th>
      <td>278854</td>
      <td>portland, oregon, usa</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>278854</th>
      <td>278855</td>
      <td>tacoma, washington, united kingdom</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>278855</th>
      <td>278856</td>
      <td>brampton, ontario, canada</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>278856</th>
      <td>278857</td>
      <td>knoxville, tennessee, usa</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>278857</th>
      <td>278858</td>
      <td>dublin, n/a, ireland</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>278858 rows × 3 columns</p>




#### Base de Ratings


```python
# Importação dos Dados Referentes aos Ratings dados aos Livros (Avaliação do Usuário em relação ao Livro)
ratings = pd.read_csv("BX-Book-Ratings.csv", sep=';', encoding="latin-1", error_bad_lines= False)
```
    


```python
ratings.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1149780 entries, 0 to 1149779
    Data columns (total 3 columns):
     #   Column       Non-Null Count    Dtype 
    ---  ------       --------------    ----- 
     0   User-ID      1149780 non-null  int64 
     1   ISBN         1149780 non-null  object
     2   Book-Rating  1149780 non-null  int64 
    dtypes: int64(2), object(1)
    memory usage: 26.3+ MB
    

# 3° Pré-Processamento dos Dados

### Renomeando Colunas

Agora, no arquivo de livros, temos algumas colunas extras que não são necessárias para nossa tarefa, como URLs de imagens. E vamos renomear as colunas de cada arquivo, pois o nome da coluna contém espaço e letras maiúsculas, então faremos as correções para facilitar o uso.


```python
# Rename de Colunas
books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']]
books.rename(columns = {'Book-Title':'title', 'Book-Author':'author', 'Year-Of-Publication':'year', 'Publisher':'publisher'}, inplace=True)
users.rename(columns = {'User-ID':'user_id', 'Location':'location', 'Age':'age'}, inplace=True)
ratings.rename(columns = {'User-ID':'user_id', 'Book-Rating':'rating'}, inplace=True)
```
    


```python
books
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ISBN</th>
      <th>title</th>
      <th>author</th>
      <th>year</th>
      <th>publisher</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0195153448</td>
      <td>Classical Mythology</td>
      <td>Mark P. O. Morford</td>
      <td>2002</td>
      <td>Oxford University Press</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0002005018</td>
      <td>Clara Callan</td>
      <td>Richard Bruce Wright</td>
      <td>2001</td>
      <td>HarperFlamingo Canada</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0060973129</td>
      <td>Decision in Normandy</td>
      <td>Carlo D'Este</td>
      <td>1991</td>
      <td>HarperPerennial</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0374157065</td>
      <td>Flu: The Story of the Great Influenza Pandemic...</td>
      <td>Gina Bari Kolata</td>
      <td>1999</td>
      <td>Farrar Straus Giroux</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0393045218</td>
      <td>The Mummies of Urumchi</td>
      <td>E. J. W. Barber</td>
      <td>1999</td>
      <td>W. W. Norton &amp;amp; Company</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>271355</th>
      <td>0440400988</td>
      <td>There's a Bat in Bunk Five</td>
      <td>Paula Danziger</td>
      <td>1988</td>
      <td>Random House Childrens Pub (Mm)</td>
    </tr>
    <tr>
      <th>271356</th>
      <td>0525447644</td>
      <td>From One to One Hundred</td>
      <td>Teri Sloat</td>
      <td>1991</td>
      <td>Dutton Books</td>
    </tr>
    <tr>
      <th>271357</th>
      <td>006008667X</td>
      <td>Lily Dale : The True Story of the Town that Ta...</td>
      <td>Christine Wicker</td>
      <td>2004</td>
      <td>HarperSanFrancisco</td>
    </tr>
    <tr>
      <th>271358</th>
      <td>0192126040</td>
      <td>Republic (World's Classics)</td>
      <td>Plato</td>
      <td>1996</td>
      <td>Oxford University Press</td>
    </tr>
    <tr>
      <th>271359</th>
      <td>0767409752</td>
      <td>A Guided Tour of Rene Descartes' Meditations o...</td>
      <td>Christopher  Biffle</td>
      <td>2000</td>
      <td>McGraw-Hill Humanities/Social Sciences/Languages</td>
    </tr>
  </tbody>
</table>
<p>271360 rows × 5 columns</p>





```python
#Quantidade de Ratings por Usuários
ratings['user_id'].value_counts()
```




    11676     13602
    198711     7550
    153662     6109
    98391      5891
    35859      5850
              ...  
    116180        1
    116166        1
    116154        1
    116137        1
    276723        1
    Name: user_id, Length: 105283, dtype: int64




```python
# Livros que tenham mais de 200 avaliações
x = ratings['user_id'].value_counts() > 200
```


```python
x
```




    11676      True
    198711     True
    153662     True
    98391      True
    35859      True
              ...  
    116180    False
    116166    False
    116154    False
    116137    False
    276723    False
    Name: user_id, Length: 105283, dtype: bool




```python
# Quantidade Usuários
# user_ids
y = x[x].index  
print(y.shape)
```

    (899,)
    


```python
y
```




    Int64Index([ 11676, 198711, 153662,  98391,  35859, 212898, 278418,  76352,
                110973, 235105,
                ...
                260183,  73681,  44296, 155916,   9856, 274808,  28634,  59727,
                268622, 188951],
               dtype='int64', length=899)



#### *Decisão de Negócio*


```python
# Trazendo ratings somente dos usuários q avaliaram mais de 200 livros
ratings = ratings[ratings['user_id'].isin(y)]
```


```python
ratings
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>ISBN</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1456</th>
      <td>277427</td>
      <td>002542730X</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>277427</td>
      <td>0026217457</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1458</th>
      <td>277427</td>
      <td>003008685X</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1459</th>
      <td>277427</td>
      <td>0030615321</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1460</th>
      <td>277427</td>
      <td>0060002050</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1147612</th>
      <td>275970</td>
      <td>3829021860</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1147613</th>
      <td>275970</td>
      <td>4770019572</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1147614</th>
      <td>275970</td>
      <td>896086097</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1147615</th>
      <td>275970</td>
      <td>9626340762</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1147616</th>
      <td>275970</td>
      <td>9626344990</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>526356 rows × 3 columns</p>





```python
# Juntando tabelas (Join ou Merge)
rating_with_books = ratings.merge(books, on='ISBN')
rating_with_books.head()
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>ISBN</th>
      <th>rating</th>
      <th>title</th>
      <th>author</th>
      <th>year</th>
      <th>publisher</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>277427</td>
      <td>002542730X</td>
      <td>10</td>
      <td>Politically Correct Bedtime Stories: Modern Ta...</td>
      <td>James Finn Garner</td>
      <td>1994</td>
      <td>John Wiley &amp;amp; Sons Inc</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3363</td>
      <td>002542730X</td>
      <td>0</td>
      <td>Politically Correct Bedtime Stories: Modern Ta...</td>
      <td>James Finn Garner</td>
      <td>1994</td>
      <td>John Wiley &amp;amp; Sons Inc</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11676</td>
      <td>002542730X</td>
      <td>6</td>
      <td>Politically Correct Bedtime Stories: Modern Ta...</td>
      <td>James Finn Garner</td>
      <td>1994</td>
      <td>John Wiley &amp;amp; Sons Inc</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12538</td>
      <td>002542730X</td>
      <td>10</td>
      <td>Politically Correct Bedtime Stories: Modern Ta...</td>
      <td>James Finn Garner</td>
      <td>1994</td>
      <td>John Wiley &amp;amp; Sons Inc</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13552</td>
      <td>002542730X</td>
      <td>0</td>
      <td>Politically Correct Bedtime Stories: Modern Ta...</td>
      <td>James Finn Garner</td>
      <td>1994</td>
      <td>John Wiley &amp;amp; Sons Inc</td>
    </tr>
  </tbody>
</table>





```python
#Quantidade de rating dos livros
number_rating = rating_with_books.groupby('title')['rating'].count().reset_index()

```


```python
number_rating
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A Light in the Storm: The Civil War Diary of ...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Always Have Popsicles</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Apple Magic (The Collector's series)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Beyond IBM: Leadership Marketing and Finance ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Clifford Visita El Hospital (Clifford El Gran...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>160264</th>
      <td>Ã?Â?ber die Pflicht zum Ungehorsam gegen den S...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>160265</th>
      <td>Ã?Â?lpiraten.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>160266</th>
      <td>Ã?Â?rger mit Produkt X. Roman.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>160267</th>
      <td>Ã?Â?stlich der Berge.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>160268</th>
      <td>Ã?Â?thique en toc</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>160269 rows × 2 columns</p>





```python
#Renomeando coluna
number_rating.rename(columns= {'rating':'number_of_ratings'}, inplace=True)
number_rating

```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>number_of_ratings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A Light in the Storm: The Civil War Diary of ...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Always Have Popsicles</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Apple Magic (The Collector's series)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Beyond IBM: Leadership Marketing and Finance ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Clifford Visita El Hospital (Clifford El Gran...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>160264</th>
      <td>Ã?Â?ber die Pflicht zum Ungehorsam gegen den S...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>160265</th>
      <td>Ã?Â?lpiraten.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>160266</th>
      <td>Ã?Â?rger mit Produkt X. Roman.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>160267</th>
      <td>Ã?Â?stlich der Berge.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>160268</th>
      <td>Ã?Â?thique en toc</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>160269 rows × 2 columns</p>





```python
# Juntando a tabela de livros com os Ratings com a tabela de quantidade de ratings por livro
final_rating = rating_with_books.merge(number_rating, on='title')
final_rating
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>ISBN</th>
      <th>rating</th>
      <th>title</th>
      <th>author</th>
      <th>year</th>
      <th>publisher</th>
      <th>number_of_ratings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>277427</td>
      <td>002542730X</td>
      <td>10</td>
      <td>Politically Correct Bedtime Stories: Modern Ta...</td>
      <td>James Finn Garner</td>
      <td>1994</td>
      <td>John Wiley &amp;amp; Sons Inc</td>
      <td>82</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3363</td>
      <td>002542730X</td>
      <td>0</td>
      <td>Politically Correct Bedtime Stories: Modern Ta...</td>
      <td>James Finn Garner</td>
      <td>1994</td>
      <td>John Wiley &amp;amp; Sons Inc</td>
      <td>82</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11676</td>
      <td>002542730X</td>
      <td>6</td>
      <td>Politically Correct Bedtime Stories: Modern Ta...</td>
      <td>James Finn Garner</td>
      <td>1994</td>
      <td>John Wiley &amp;amp; Sons Inc</td>
      <td>82</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12538</td>
      <td>002542730X</td>
      <td>10</td>
      <td>Politically Correct Bedtime Stories: Modern Ta...</td>
      <td>James Finn Garner</td>
      <td>1994</td>
      <td>John Wiley &amp;amp; Sons Inc</td>
      <td>82</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13552</td>
      <td>002542730X</td>
      <td>0</td>
      <td>Politically Correct Bedtime Stories: Modern Ta...</td>
      <td>James Finn Garner</td>
      <td>1994</td>
      <td>John Wiley &amp;amp; Sons Inc</td>
      <td>82</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>487666</th>
      <td>275970</td>
      <td>1892145022</td>
      <td>0</td>
      <td>Here Is New York</td>
      <td>E. B. White</td>
      <td>1999</td>
      <td>Little Bookroom</td>
      <td>1</td>
    </tr>
    <tr>
      <th>487667</th>
      <td>275970</td>
      <td>1931868123</td>
      <td>0</td>
      <td>There's a Porcupine in My Outhouse: Misadventu...</td>
      <td>Mike Tougias</td>
      <td>2002</td>
      <td>Capital Books (VA)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>487668</th>
      <td>275970</td>
      <td>3411086211</td>
      <td>10</td>
      <td>Die Biene.</td>
      <td>Sybil GrÃ?Â¤fin SchÃ?Â¶nfeldt</td>
      <td>1993</td>
      <td>Bibliographisches Institut, Mannheim</td>
      <td>1</td>
    </tr>
    <tr>
      <th>487669</th>
      <td>275970</td>
      <td>3829021860</td>
      <td>0</td>
      <td>The Penis Book</td>
      <td>Joseph Cohen</td>
      <td>1999</td>
      <td>Konemann</td>
      <td>1</td>
    </tr>
    <tr>
      <th>487670</th>
      <td>275970</td>
      <td>4770019572</td>
      <td>0</td>
      <td>Musashi</td>
      <td>Eiji Yoshikawa</td>
      <td>1995</td>
      <td>Kodansha International (JPN)</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>487671 rows × 8 columns</p>




#### *Decisão de Negócio*


```python
# Filtrar somente livros que tenham pelo menos 50 avaliações
final_rating = final_rating[final_rating['number_of_ratings'] >= 50]
final_rating.shape
```




    (61853, 8)




```python
# Vamos descartar os valores duplicados, porque se o mesmo usuário tiver avaliado o mesmo livro várias vezes, isso pode dar rúim.
final_rating.drop_duplicates(['user_id','title'], inplace=True)
final_rating.shape
```
    




    (59850, 8)




```python
final_rating
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>ISBN</th>
      <th>rating</th>
      <th>title</th>
      <th>author</th>
      <th>year</th>
      <th>publisher</th>
      <th>number_of_ratings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>277427</td>
      <td>002542730X</td>
      <td>10</td>
      <td>Politically Correct Bedtime Stories: Modern Ta...</td>
      <td>James Finn Garner</td>
      <td>1994</td>
      <td>John Wiley &amp;amp; Sons Inc</td>
      <td>82</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3363</td>
      <td>002542730X</td>
      <td>0</td>
      <td>Politically Correct Bedtime Stories: Modern Ta...</td>
      <td>James Finn Garner</td>
      <td>1994</td>
      <td>John Wiley &amp;amp; Sons Inc</td>
      <td>82</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11676</td>
      <td>002542730X</td>
      <td>6</td>
      <td>Politically Correct Bedtime Stories: Modern Ta...</td>
      <td>James Finn Garner</td>
      <td>1994</td>
      <td>John Wiley &amp;amp; Sons Inc</td>
      <td>82</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12538</td>
      <td>002542730X</td>
      <td>10</td>
      <td>Politically Correct Bedtime Stories: Modern Ta...</td>
      <td>James Finn Garner</td>
      <td>1994</td>
      <td>John Wiley &amp;amp; Sons Inc</td>
      <td>82</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13552</td>
      <td>002542730X</td>
      <td>0</td>
      <td>Politically Correct Bedtime Stories: Modern Ta...</td>
      <td>James Finn Garner</td>
      <td>1994</td>
      <td>John Wiley &amp;amp; Sons Inc</td>
      <td>82</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>236701</th>
      <td>255489</td>
      <td>0553579983</td>
      <td>7</td>
      <td>And Then You Die</td>
      <td>Iris Johansen</td>
      <td>1998</td>
      <td>Bantam</td>
      <td>50</td>
    </tr>
    <tr>
      <th>236702</th>
      <td>256407</td>
      <td>0553579983</td>
      <td>0</td>
      <td>And Then You Die</td>
      <td>Iris Johansen</td>
      <td>1998</td>
      <td>Bantam</td>
      <td>50</td>
    </tr>
    <tr>
      <th>236703</th>
      <td>257204</td>
      <td>0553579983</td>
      <td>0</td>
      <td>And Then You Die</td>
      <td>Iris Johansen</td>
      <td>1998</td>
      <td>Bantam</td>
      <td>50</td>
    </tr>
    <tr>
      <th>236704</th>
      <td>261829</td>
      <td>0553579983</td>
      <td>0</td>
      <td>And Then You Die</td>
      <td>Iris Johansen</td>
      <td>1998</td>
      <td>Bantam</td>
      <td>50</td>
    </tr>
    <tr>
      <th>236705</th>
      <td>273979</td>
      <td>0553579983</td>
      <td>0</td>
      <td>And Then You Die</td>
      <td>Iris Johansen</td>
      <td>1998</td>
      <td>Bantam</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
<p>59850 rows × 8 columns</p>




### Vamos fazer uma parada que é o seguinte:

Vamos transpor os **usuários** em **colunas**, ao invés de linhas, pois as avaliações dadas por eles serão as **variáveis** da máquina preditiva.


```python
final_rating.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 59850 entries, 0 to 236705
    Data columns (total 8 columns):
     #   Column             Non-Null Count  Dtype 
    ---  ------             --------------  ----- 
     0   user_id            59850 non-null  int64 
     1   ISBN               59850 non-null  object
     2   rating             59850 non-null  int64 
     3   title              59850 non-null  object
     4   author             59850 non-null  object
     5   year               59850 non-null  object
     6   publisher          59850 non-null  object
     7   number_of_ratings  59850 non-null  int64 
    dtypes: int64(3), object(5)
    memory usage: 4.1+ MB
    


```python
# Transposição de linhas(users_id) em colunas
book_pivot = final_rating.pivot_table(columns='user_id', index='title', values="rating")
```


```python
book_pivot
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>user_id</th>
      <th>254</th>
      <th>2276</th>
      <th>2766</th>
      <th>2977</th>
      <th>3363</th>
      <th>3757</th>
      <th>4017</th>
      <th>4385</th>
      <th>6242</th>
      <th>6251</th>
      <th>...</th>
      <th>274004</th>
      <th>274061</th>
      <th>274301</th>
      <th>274308</th>
      <th>274808</th>
      <th>275970</th>
      <th>277427</th>
      <th>277478</th>
      <th>277639</th>
      <th>278418</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1984</th>
      <td>9.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1st to Die: A Novel</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2nd Chance</th>
      <td>NaN</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4 Blondes</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>84 Charing Cross Road</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Year of Wonders</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>You Belong To Me</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Zen and the Art of Motorcycle Maintenance: An Inquiry into Values</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Zoya</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>\O\" Is for Outlaw"</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>742 rows × 888 columns</p>





```python
book_pivot.shape
```




    (742, 888)




```python
book_pivot.fillna(0, inplace=True)
```


```python
book_pivot
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>user_id</th>
      <th>254</th>
      <th>2276</th>
      <th>2766</th>
      <th>2977</th>
      <th>3363</th>
      <th>3757</th>
      <th>4017</th>
      <th>4385</th>
      <th>6242</th>
      <th>6251</th>
      <th>...</th>
      <th>274004</th>
      <th>274061</th>
      <th>274301</th>
      <th>274308</th>
      <th>274808</th>
      <th>275970</th>
      <th>277427</th>
      <th>277478</th>
      <th>277639</th>
      <th>278418</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1984</th>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1st to Die: A Novel</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2nd Chance</th>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4 Blondes</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>84 Charing Cross Road</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Year of Wonders</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>You Belong To Me</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Zen and the Art of Motorcycle Maintenance: An Inquiry into Values</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Zoya</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>\O\" Is for Outlaw"</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>742 rows × 888 columns</p>




Preparamos nosso conjunto de dados para modelagem. Usaremos o algoritmo de vizinhos mais próximos (nearest neighbors algorithm), que é usado para agrupamento com base na **distância euclidiana**.



**Nesta aula explicadim**:

https://www.youtube.com/watch?v=jD4AKp4-Tmo





Mas aqui na tabela dinâmica, temos muitos valores zero e no agrupamento, esse poder de computação aumentará para calcular a distância dos valores zero, portanto, converteremos a tabela dinâmica para a matriz esparsa e, em seguida, alimentaremos o modelo.


```python
from scipy.sparse import csr_matrix
book_sparse = csr_matrix(book_pivot)
```

#4° Criação da Máquina Preditiva

https://scikit-learn.org/stable/modules/neighbors.html


```python
from sklearn.neighbors import NearestNeighbors
model = NearestNeighbors(algorithm='brute')
model.fit(book_sparse)
```



## Novas Predições


```python
#1984
distances, suggestions = model.kneighbors(book_pivot.iloc[0, :].values.reshape(1, -1))
```


```python
book_pivot.head()
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>user_id</th>
      <th>254</th>
      <th>2276</th>
      <th>2766</th>
      <th>2977</th>
      <th>3363</th>
      <th>3757</th>
      <th>4017</th>
      <th>4385</th>
      <th>6242</th>
      <th>6251</th>
      <th>...</th>
      <th>274004</th>
      <th>274061</th>
      <th>274301</th>
      <th>274308</th>
      <th>274808</th>
      <th>275970</th>
      <th>277427</th>
      <th>277478</th>
      <th>277639</th>
      <th>278418</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1984</th>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1st to Die: A Novel</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2nd Chance</th>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4 Blondes</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>84 Charing Cross Road</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 888 columns</p>





```python
for i in range(len(suggestions)):
  print(book_pivot.index[suggestions[i]])
```

    Index(['1984', 'No Safe Place', 'A Civil Action', 'Foucault's Pendulum',
           'Long After Midnight'],
          dtype='object', name='title')
    


```python
#Hannibal
distances, suggestions = model.kneighbors(book_pivot.iloc[236, :].values.reshape(1, -1))
```


```python
book_pivot.head(236)
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>user_id</th>
      <th>254</th>
      <th>2276</th>
      <th>2766</th>
      <th>2977</th>
      <th>3363</th>
      <th>3757</th>
      <th>4017</th>
      <th>4385</th>
      <th>6242</th>
      <th>6251</th>
      <th>...</th>
      <th>274004</th>
      <th>274061</th>
      <th>274301</th>
      <th>274308</th>
      <th>274808</th>
      <th>275970</th>
      <th>277427</th>
      <th>277478</th>
      <th>277639</th>
      <th>278418</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1984</th>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1st to Die: A Novel</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2nd Chance</th>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4 Blondes</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>84 Charing Cross Road</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Guardian Angel</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Guilty Pleasures (Anita Blake Vampire Hunter (Paperback))</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Guilty as Sin</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>H Is for Homicide (Kinsey Millhone Mysteries (Paperback))</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Hannibal</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>236 rows × 888 columns</p>





```python
for i in range(len(suggestions)):
  print(book_pivot.index[suggestions[i]])
```

    Index(['Hard Eight : A Stephanie Plum Novel (A Stephanie Plum Novel)',
           'Seven Up (A Stephanie Plum Novel)',
           'Hot Six : A Stephanie Plum Novel (A Stephanie Plum Novel)',
           'The Next Accident', 'The Mulberry Tree'],
          dtype='object', name='title')
    


```python
#Harry Potter
distances, suggestions = model.kneighbors(book_pivot.iloc[238, :].values.reshape(1, -1))
```


```python
book_pivot.head(238)
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>user_id</th>
      <th>254</th>
      <th>2276</th>
      <th>2766</th>
      <th>2977</th>
      <th>3363</th>
      <th>3757</th>
      <th>4017</th>
      <th>4385</th>
      <th>6242</th>
      <th>6251</th>
      <th>...</th>
      <th>274004</th>
      <th>274061</th>
      <th>274301</th>
      <th>274308</th>
      <th>274808</th>
      <th>275970</th>
      <th>277427</th>
      <th>277478</th>
      <th>277639</th>
      <th>278418</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1984</th>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1st to Die: A Novel</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2nd Chance</th>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4 Blondes</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>84 Charing Cross Road</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Guilty as Sin</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>H Is for Homicide (Kinsey Millhone Mysteries (Paperback))</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Hannibal</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Hard Eight : A Stephanie Plum Novel (A Stephanie Plum Novel)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Harry Potter and the Chamber of Secrets (Book 2)</th>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>238 rows × 888 columns</p>





```python
for i in range(len(suggestions)):
  print(book_pivot.index[suggestions[i]])
```

    Index(['Harry Potter and the Goblet of Fire (Book 4)',
           'Harry Potter and the Prisoner of Azkaban (Book 3)',
           'Harry Potter and the Order of the Phoenix (Book 5)',
           'The Cradle Will Fall', 'Exclusive'],
          dtype='object', name='title')
    

# Fim

## Valeu!

Fonte de Inspiração:

https://www.analyticsvidhya.com/blog/2021/06/build-book-recommendation-system-unsupervised-learning-project/
