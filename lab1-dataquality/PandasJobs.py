import pandas as pd  
import re

# charger le dataset
jobs_df = pd.read_csv('jobs.csv', sep=';')

# verification des valeurs manquantes
def is_problematic(column):
    sum_of_values = len(jobs_df)
    sum_of_null_values = jobs_df[column].isnull().sum()
    if sum_of_values > 0 and sum_of_null_values / sum_of_values > 0.8:
        return True
    return False

for column in jobs_df.columns:
    if is_problematic(column):
        print(f"{column} a plus de 80% de valeurs manquantes")

# permet de verifier si une colonne est une date (propore a ce dataset)
def is_column_of_dates(column):
    date_pattern = re.compile(r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2} AM')
    date_pattern2 = re.compile(r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2} PM')
    date_pattern3 = re.compile(r'\d{2}/\d{2}/\d{4}')
    for value in column:
        if not date_pattern.match(str(value)) and not date_pattern2.match(str(value)) and not date_pattern3.match(str(value)):
            return False
    return True

# verification des dates
for column in jobs_df.columns:
    if is_column_of_dates(jobs_df[column]):
        for value in jobs_df[column]:
            if not re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', str(value)) and not re.match(r'\d{4}-\d{2}-\d{2}', str(value)):
                print(f"{column} n'est pas une date valide dans le format attendu YYYY-MM-DD HH:MM:SS ou YYYY-MM-DD")
                break
            
# supprimer les colonnes problematiques
for column in jobs_df.columns:
    if is_problematic(column):
        # supprimer la colonne si elle a plus de 80% de valeurs manquantes
        jobs_df.drop(column, axis=1, inplace=True) 

# remplacer les valeurs manquantes
for column in jobs_df.columns:
    if jobs_df[column].dtype == 'float64' or jobs_df[column].dtype == 'int64':
        # remplace les valeurs manquantes par la moyenne
        jobs_df[column].fillna(jobs_df[column].mean(), inplace=True)
    elif jobs_df[column].dtype == 'object':
        # remplacer les valeurs manquantes par la valeur la plus frequente
        jobs_df[column].fillna(jobs_df[column].mode()[0], inplace=True)

# mettre les dates dans le bon format 
for column in jobs_df.columns:
    # verifier si la colonne est une colonne de dates
    if is_column_of_dates(jobs_df[column]):
        for value in jobs_df[column]:
            # verifier si la date est dans le bon format
            if not re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', str(value)) and not re.match(r'\d{4}-\d{2}-\d{2}', str(value)):
                # verifier si la date est dans un certain format (MM/DD/YYYY HH:MM:SS AM/PM) et la convertir en YYYY-MM-DD HH:MM:SS pour ce dataset 
                if re.match(r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2} AM', str(value)) or re.match(r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2} PM', str(value)):
                    time = value.split(' ')[1]
                    if 'AM' in value:
                        if time.startswith('12'):
                            time = time.replace('12', '00')
                    elif 'PM' in value:
                        hour, minute, second = time.split(':')
                        if hour != '12':
                            hour = str(int(hour) + 12)
                        time = f"{hour}:{minute}:{second}"
                    time = time.replace(' AM', '').replace(' PM', '')
                    date = value.split(' ')[0]
                    date = pd.to_datetime(date).strftime('%Y-%m-%d')
                    new_value = f"{date} {time}"
                    jobs_df[column] = jobs_df[column].replace(value, new_value)
                # verifier si la date est dans un certain format (MM/DD/YYYY) et la convertir en YYYY-MM-DD pour ce dataset
                elif re.match(r'\d{2}/\d{2}/\d{4}', str(value)):
                    jobs_df[column] = pd.to_datetime(jobs_df[column]).dt.strftime('%Y-%m-%d')
  

# sauvegarder le nouveau dataframe nettoye
jobs_df.to_csv('cleaned_jobs.csv', sep=';', index=False)