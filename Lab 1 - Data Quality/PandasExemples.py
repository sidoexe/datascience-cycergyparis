#Programme venu de https://www.learndatasci.com/tutorials/python-pandas-tutorial-complete-introduction-for-beginners/
import pandas as pd

movies_df = pd.read_csv('IMDB-Movie-Data.csv', index_col="Title")
print(movies_df.head()) # affiche les 5 premieres lignes du dataframe
print(movies_df.info()) # affiche les colonnes et les types de données du dataframe

print("dataset=" + str(movies_df.shape)) # affiche le nombre de lignes et de colonnes du dataframe sous forme de chaine de caracteres
print("dataset=" , movies_df.shape) # affiche le nombre de lignes et de colonnes du dataframe 

# renommer les colonnes du dataframe en minuscule
movies_df.columns = [col.lower() for col in movies_df]
print("Columns=" ,movies_df.columns)

# j'ai remarqué que le nom de la colonne metascore est mal orthographié, je l'ai donc corrigé
print(movies_df['metascore'].value_counts()) # affiche le nombre de redoondance de chaque valeur de la colonne metascore avec les valeurs nulles
print(movies_df['metascore'].value_counts(dropna=False))  # affiche le nombre de redoondance de chaque valeur de la colonne metascore sans les valeurs nulles

print("\n--- origine ---")
print(movies_df.info())
print("\n--- Drop 0 ---")
temp_df=movies_df.dropna() # supprime les lignes contenant des valeurs nulles 
print(temp_df.info()) # baisse du memory usage de 12 KB 
print("\n--- Drop 1 ---")
temp_df=movies_df.dropna(axis=1) # pour supprimer les colonnes contenant des valeurs nulles, on doit specifier l'axe de suppression 
print(temp_df.info())

print("\n--- origine ---")
print(movies_df.info())

revenue = movies_df['revenue (millions)'] # recupere la colonne revenue (millions) du dataframe (j'ai changé le nom de la colonne en revenue (millions) car on a tout mis en minuscule en haut)
revenue_mean = revenue.mean() # calcule la moyenne des valeurs de la colonne revenue (millions)
temp_df=movies_df.fillna(value={'Revenue (Millions)':revenue_mean}) # remplace les valeurs nulles de la colonne revenue (millions) par la moyenne des valeurs de cette colonne
print("\n--- --  ---")
print(temp_df.isnull().sum()) # affiche le nombre de valeurs nulles de chaque colonne du dataframe
print("\n--- Fill  ---")
print(temp_df.info())


print("\n------- Descrition ---------")
print(temp_df.describe()) # affiche les statistiques descriptives des colonnes numeriques du dataframe

print("\n", movies_df['genre'].describe()) # j'ai changé le nom de la colonne en genre car on a tout mis en minuscule en haut
print("\n",movies_df['genre'].value_counts().head(10))
