import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os



################################################## ZADANIE 1 ##################################################

data_path = './data'
names_list = []
for filename in glob.glob(os.path.join(data_path, '*.txt')):
    year = filename[-8:-4]
    data = pd.read_csv(filename, delimiter=',', header=None)
    data = data.assign(year=year)
    names_list.append(data)


names_df = pd.concat(names_list, ignore_index=True)
names_df.columns = ['Name', 'Sex', 'Number', 'Year']



################################################## ZADANIE 2 ##################################################

number_of_unique_names = names_df['Name'].nunique()

print(f'Odpowiedz do zadania 2.:')
print(f'W tym czasie nadano {number_of_unique_names} roznych imion.\n')



################################################## ZADANIE 3 ##################################################

male_names_df = names_df.loc[names_df['Sex'] == 'M']
number_of_unique_male_names = male_names_df['Name'].nunique()

female_names_df = names_df.loc[names_df['Sex'] == 'F']
number_of_unique_female_names = female_names_df['Name'].nunique()

print(f'Odpowiedz do zadania 3.:')
print(f'W tym czasie nadano {number_of_unique_male_names} roznych imion meskich'
      f'i {number_of_unique_female_names} zenskich.\n')



################################################## ZADANIE 4 ##################################################

def count_sex_frequency(df):
    df['Total number'] = df.groupby(['Year', 'Sex'])['Number'].transform('sum')

    df['frequency_male'] = np.where(
        df['Sex'] == 'M',
        df['Number'] / df['Total number'],
        0
    )
    df['frequency_female'] = np.where(
        df['Sex'] == 'F',
        df['Number'] / df['Total number'],
        0
    )

    return df


names_df = count_sex_frequency(names_df)


################################################## ZADANIE 5 ##################################################

births_by_year_and_sex = names_df.groupby(['Year', 'Sex'])['Number'].sum().unstack(fill_value=0)
births_by_year_and_sex['Total'] = births_by_year_and_sex.sum(axis=1)

years = births_by_year_and_sex.index.values
total_births = births_by_year_and_sex['Total'].values
female_to_male_ratio = births_by_year_and_sex['F'] / births_by_year_and_sex['M']

min_diff_year = years[np.abs(female_to_male_ratio - 1).argmin()]
max_diff_year = years[np.abs(female_to_male_ratio - 1).argmax()]

print(f'Odpowiedz do zadania 5.:')
print(f'Najmniejsza roznica w liczbie urodzen byla w roku {min_diff_year}, a najwieksza w {max_diff_year}.\n')

tick_size = 5
fig, axs = plt.subplots(2, 1, figsize=(14, 8))

axs[0].plot(years, total_births, label='Łączna liczba urodzeń')
axs[0].set_xticks(years[::tick_size])
axs[0].tick_params(labelrotation=90, labelsize=8)
axs[0].grid(visible=True, which='both')
axs[0].set_title('Łączna liczba urodzeń w poszczególnych latach')
axs[0].legend()

axs[1].plot(years, female_to_male_ratio, label='Stosunek urodzeń dziewcząt do chłopców')
axs[1].set_xticks(years[::tick_size])
axs[1].tick_params(labelrotation=90, labelsize=8)
axs[1].grid(visible=True, which='both')
axs[1].set_title('Stosunek urodzeń dziewcząt do chłopców w poszczególnych latach')
axs[1].legend()

plt.tight_layout()
plt.show()



################################################## ZADANIE 6 ##################################################

def top_x_names_ranking(df, number):
    names_frequency_df = df.groupby(['Name', 'Sex'])[['frequency_female', 'frequency_male']].sum().reset_index()

    female_names = names_frequency_df[names_frequency_df['Sex'] == 'F']
    male_names = names_frequency_df[names_frequency_df['Sex'] == 'M']

    female_top_x = (female_names.nlargest(number, 'frequency_female')['Name'].reset_index(drop=True))
    male_top_x = (male_names.nlargest(number, 'frequency_male')['Name'].reset_index(drop=True))

    return female_top_x, male_top_x


female_top1000_names, male_top1000_names = top_x_names_ranking(names_df, 1000)



################################################## ZADANIE 7 ##################################################

male_name = 'John'
female_name = female_top1000_names.iloc[0]
years_to_check = ['1934', '1980', '2022']

male_name_data = names_df[(names_df['Name'] == male_name) & (names_df['Sex'] == 'M')]
female_name_data = names_df[(names_df['Name'] == female_name) & (names_df['Sex'] == 'F')]

male_counts = male_name_data[male_name_data['Year'].isin(years_to_check)]['Number'].values
female_counts = female_name_data[female_name_data['Year'].isin(years_to_check)]['Number'].values

male_name_data = male_name_data.sort_values(by="Year", ascending=True)
female_name_data = female_name_data.sort_values(by="Year", ascending=True)

fig, ax1 = plt.subplots(figsize=(14, 8))

ax1.plot(male_name_data['Year'], male_name_data['Number'], label=f'Liczba imienia {male_name}')
ax1.plot(female_name_data['Year'], female_name_data['Number'], label=f'Liczba imienia {female_name}')
ax1.set_ylabel('Liczba nadanych imion')
ax1.tick_params(axis='y')
ax1.set_xticks(male_name_data['Year'][::tick_size])
ax1.tick_params(axis='x', rotation=90)

for year, count in zip(years_to_check, male_counts):
    x = year
    y = int(male_name_data[male_name_data['Year'] == str(year)]['Number'].values[0])
    ax1.plot(x, y, 'o', markersize=3, color='black')
    ax1.annotate(str(count), (x, y), (x, y+1000), color='black', weight='bold', ha='center', fontsize=10)

for year, count in zip(years_to_check, female_counts):
    x = year
    y = int(female_name_data[female_name_data['Year'] == str(year)]['Number'].values[0])
    ax1.plot(x, y, 'o', markersize=3, color='black')
    ax1.annotate(str(count), (x, y), (x, y + 1000), color='black', weight='bold', ha='center', fontsize=10)

ax2 = ax1.twinx()
ax2.plot(male_name_data['Year'], male_name_data['frequency_male'],
         label=f'Popularność {male_name}', linestyle='--')
ax2.plot(female_name_data['Year'], female_name_data['frequency_female'],
         label=f'Popularność {female_name}', linestyle='--')
ax2.set_ylabel('Popularność (częstotliwość)', color='black')
ax2.tick_params(axis='y', labelcolor='black')

fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

plt.title('Zmiany liczby i popularności imion w czasie')
plt.grid(visible=True, which='both')
plt.tight_layout()
plt.show()



################################################## ZADANIE 8 ##################################################

def percent_of_names_in_year(df, female_top_names, male_top_names):
    df['is_top_x'] = df['Name'].isin(female_top_names) | df['Name'].isin(male_top_names)

    total_names_by_year_sex = df.groupby(['Year', 'Sex'])['Number'].sum()
    top_names_by_year_sex = df[df['is_top_x']].groupby(['Year', 'Sex'])['Number'].sum()

    top_percentage = top_names_by_year_sex / total_names_by_year_sex * 100

    female_percentage = top_percentage.xs('F', level='Sex')
    male_percentage = top_percentage.xs('M', level='Sex')

    difference = abs(female_percentage - male_percentage)
    year_of_max_difference = difference.idxmax()

    return top_percentage, year_of_max_difference


top1000_percentage, max_diff_year_top1000 = percent_of_names_in_year(names_df, female_top1000_names, male_top1000_names)

print(f'Odpowiedz do zadania 8.:')
print(f'{max_diff_year_top1000} to rok, w ktorym zaobserwowano najwieksza roznice w roznorodnosci'
      f' miedzy imionami meskimi a zenskimi.')
print(f'Na przestrzeni ostatnich 140 lat różnorodność imion zdecydowanie zwiększyła się i w przypadku każdego roku'
      f' można zauważyć, że imiona żeńskie były bardziej różnorodne, jeżeli chodzi o porównanie do imion męskich.\n')

fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(top1000_percentage.xs('M', level='Sex'), label='Mężczyźni')
ax.plot(top1000_percentage.xs('F', level='Sex'), label='Kobiety')
ax.axvline(max_diff_year_top1000, color='black', linestyle='--',
           label=f'Rok największej różnicy w różnorodności imion ({max_diff_year_top1000})')
ax.set_xticks(years[::tick_size])
ax.tick_params(labelrotation=90, labelsize=8)
ax.set_ylabel("Procent imion z top1000")
ax.set_xlabel("Rok")
ax.set_title("Zmiana różnorodności imion w czasie z podziałem na płeć")
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()



################################################## ZADANIE 9 ##################################################

names_df['Last Letter'] = names_df['Name'].str[-1]

last_letter_data = names_df.groupby(['Year', 'Sex', 'Last Letter'])['Number'].sum().reset_index()
last_letter_data['Total'] = last_letter_data.groupby(['Year', 'Sex'])['Number'].transform('sum')
last_letter_data['Frequency'] = last_letter_data['Number'] / last_letter_data['Total']

selected_years = ['1910', '1970', '2023']
male_last_letter_data = last_letter_data[(last_letter_data['Sex'] == 'M') &
                                         (last_letter_data['Year'].isin(selected_years))]
male_last_letter_pivot = male_last_letter_data.pivot(index='Last Letter', columns='Year', values='Frequency').fillna(0)

male_last_letter_pivot.plot(kind='bar', figsize=(14, 8))
plt.title('Popularność ostatnich liter w imionach męskich (1910, 1970, 2023)')
plt.ylabel('Częstotliwość')
plt.xlabel('Ostatnia litera')
plt.legend(title='Rok')
plt.grid(axis='both')

plt.tight_layout()
plt.show()

year1 = '1910'
year2 = '2023'
male_last_letter_pivot['Change'] = male_last_letter_pivot[year2] - male_last_letter_pivot[year1]
max_increase_letter = male_last_letter_pivot['Change'].idxmax()
max_decrease_letter = male_last_letter_pivot['Change'].idxmin()

change_max = 100 * male_last_letter_pivot['Change'].max()
change_min = 100 * male_last_letter_pivot['Change'].min()


print(f'Odpowiedz do zadania 9.:')
print(f'Litera z największym wzrostem: {max_increase_letter} ({change_max:.2f}%)')
print(f'Litera z największym spadkiem: {max_decrease_letter} ({change_min:.2f}%)\n')

top_changes_letters = male_last_letter_pivot['Change'].abs().nlargest(3).index.tolist()
trend_data = last_letter_data[(last_letter_data['Sex'] == 'M') &
                              (last_letter_data['Last Letter'].isin(top_changes_letters))]

fig, ax = plt.subplots(figsize=(14, 8))
for letter in top_changes_letters:
    letter_trend = trend_data[trend_data['Last Letter'] == letter]
    ax.plot(letter_trend['Year'], letter_trend['Frequency'], label=f'Ostatnia litera: {letter}')
    ax.set_xticks(letter_trend['Year'][::tick_size])
    ax.tick_params(labelrotation=90, labelsize=8)

plt.title('Trend popularności wybranych ostatnich liter w imionach męskich')
plt.ylabel('Częstotliwość')
plt.xlabel('Rok')
plt.legend()
plt.grid(axis='both')

plt.tight_layout()
plt.show()



################################################## ZADANIE 10 ##################################################

top1000_names_df = names_df[names_df['is_top_x']]
top1000_names_df = top1000_names_df.copy()
top1000_names_df['Year'] = pd.to_numeric(top1000_names_df['Year'], errors='coerce')

grouped = top1000_names_df.groupby(['Name', 'Year', 'Sex'])['Number'].sum().reset_index()
grouped['Total'] = grouped.groupby(['Name', 'Year'])['Number'].transform('sum')

grouped['p_m'] = (grouped['Sex'] == 'M') * (grouped['Number'] / grouped['Total'])
grouped['p_k'] = (grouped['Sex'] == 'F') * (grouped['Number'] / grouped['Total'])

pre_1920_df = grouped[grouped['Year'] <= 1920]
post_2000_df = grouped[grouped['Year'] >= 2000]
pre_1920_agg = pre_1920_df.groupby('Name').agg({'p_m': 'mean', 'p_k': 'mean'}).reset_index()
post_2000_agg = post_2000_df.groupby('Name').agg({'p_m': 'mean', 'p_k': 'mean'}).reset_index()

change_df = pd.merge(pre_1920_agg, post_2000_agg, on='Name', suffixes=('_pre', '_post'))
change_df['change'] = (change_df['p_m_post'] + change_df['p_k_pre']) / 2
change_df_sorted = change_df.sort_values('change', ascending=False)

name_m_to_f = change_df_sorted.iloc[-1]
name_f_to_m = change_df_sorted.iloc[0]

name_m_f = name_m_to_f['Name']
name_f_m = name_f_to_m['Name']

print(f'Odpowiedz do zadania 10.:')
print(f'Imię zmieniające się z męskiego na żeńskie: {name_m_f}')
print(f'Imię zmieniające się z żeńskiego na męskie: {name_f_m}\n')

years = list(range(top1000_names_df['Year'].min(), top1000_names_df['Year'].max() + 1))
genders = grouped['Sex'].unique()
full_combinations = pd.MultiIndex.from_product([grouped['Name'].unique(), years, genders],
                                               names=['Name', 'Year', 'Sex'])

full_df = pd.DataFrame(index=full_combinations).reset_index()
grouped_full = pd.merge(full_df, grouped, on=['Name', 'Year', 'Sex'], how='left')
grouped_full['p_m'] = grouped_full['p_m'].fillna(0)
grouped_full['p_k'] = grouped_full['p_k'].fillna(0)

name_m_to_f_data = grouped_full[grouped_full['Name'] == name_m_to_f['Name']]
name_f_to_m_data = grouped_full[grouped_full['Name'] == name_f_to_m['Name']]

fig, axs = plt.subplots(2, 1, figsize=(14, 8))
x_mm = name_m_to_f_data[name_m_to_f_data['Sex'] == 'M']['Year']
x_mf = name_m_to_f_data[name_m_to_f_data['Sex'] == 'F']['Year']
x_fm = name_f_to_m_data[name_f_to_m_data['Sex'] == 'M']['Year']
x_ff = name_f_to_m_data[name_f_to_m_data['Sex'] == 'F']['Year']

y_mm = name_m_to_f_data[name_m_to_f_data['Sex'] == 'M']['p_m']
y_mf = name_m_to_f_data[name_m_to_f_data['Sex'] == 'F']['p_k']
y_fm = name_f_to_m_data[name_f_to_m_data['Sex'] == 'M']['p_m']
y_ff = name_f_to_m_data[name_f_to_m_data['Sex'] == 'F']['p_k']

name_legend_m_f = name_m_to_f['Name']
name_legend_f_m = name_f_to_m['Name']

axs[0].plot(x_mm, y_mm, label='p_m (M)')
axs[0].plot(x_mf, y_mf, label='p_k (F)')
axs[0].set_xticks(years[::tick_size])
axs[0].tick_params(labelrotation=90, labelsize=8)
axs[0].grid(visible=True, which='both')
axs[0].set_title(f'Zmiana imienia {name_legend_m_f} (z męskiego na żeńskie)')
axs[0].legend()

axs[1].plot(x_fm, y_fm, label='p_m (M)')
axs[1].plot(x_ff, y_ff, label='p_k (F)')
axs[1].set_xticks(years[::tick_size])
axs[1].tick_params(labelrotation=90, labelsize=8)
axs[1].grid(visible=True, which='both')
axs[1].set_title(f'Zmiana imienia {name_legend_f_m} (z żeńskiego na męskie)\n')
axs[1].legend()

plt.tight_layout()
plt.show()



################################################## ZADANIE 11 ##################################################

##### 1 #####
conn = sqlite3.connect('./data/names_pl_2000-23.sqlite')
query = ('SELECT Imię, Płeć, Liczba, Rok FROM females UNION ALL '
         'SELECT Imię, Płeć, Liczba, Rok FROM males')
names_pl_df = pd.read_sql_query(query, conn)
conn.close()

names_pl_df.columns = ['Name', 'Sex', 'Number', 'Year']
names_pl_df['Sex'] = names_pl_df['Sex'].replace('K', 'F')
names_pl_df = names_pl_df.sort_values(by=['Year', 'Number'], ascending=[True, False]).reset_index(drop=True)


##### 2 #####
names_pl_df = count_sex_frequency(names_pl_df)
female_top200_names_pl, male_top200_names_pl = top_x_names_ranking(names_pl_df, 200)

top200_percentage, max_diff_year_top200 = percent_of_names_in_year(names_pl_df, female_top200_names_pl,
                                                                   male_top200_names_pl)

years_pl = top200_percentage.index.get_level_values('Year').unique()
tick_size_pl = 1
top1000_percentage_2000_2023 = top1000_percentage[top1000_percentage.index.get_level_values('Year') >= '2000']

fig, axs = plt.subplots(2, 1, figsize=(14, 8))

axs[0].plot(top200_percentage.xs('M', level='Sex'), label='Mężczyźni')
axs[0].plot(top200_percentage.xs('F', level='Sex'), label='Kobiety')
axs[0].set_xticks(years_pl[::tick_size_pl])
axs[0].tick_params(labelrotation=90, labelsize=8)
axs[0].set_ylabel("Procent imion z top200")
axs[0].set_xlabel("Rok")
axs[0].grid(visible=True, which='both')
axs[0].set_title("Zmiana różnorodności imion w czasie z podziałem na płeć w Polsce")
axs[0].legend()

axs[1].plot(top1000_percentage_2000_2023.xs('M', level='Sex'), label='Mężczyźni')
axs[1].plot(top1000_percentage_2000_2023.xs('F', level='Sex'), label='Kobiety')
axs[1].tick_params(labelrotation=90, labelsize=8)
axs[1].set_ylabel("Procent imion z top1000")
axs[1].set_xlabel("Rok")
axs[1].grid(visible=True, which='both')
axs[1].set_title("Zmiana różnorodności imion w czasie z podziałem na płeć w USA")
axs[1].legend()

plt.tight_layout()
plt.show()

diff_2000_2013_pl_m = (
    top200_percentage.loc[2000, "M"] - top200_percentage.loc[2013, "M"]
)
diff_2000_2013_pl_f = (
    top200_percentage.loc[2000, "F"] - top200_percentage.loc[2013, "F"]
)
diff_2000_2013_usa_m = (
    top1000_percentage_2000_2023.loc["2000", "M"] - top1000_percentage_2000_2023.loc["2013", "M"]
)
diff_2000_2013_usa_f = (
    top1000_percentage_2000_2023.loc["2000", "F"] - top1000_percentage_2000_2023.loc["2013", "F"]
)

fig, ax = plt.subplots(1, 2, figsize=(14, 8))
ax[0].hist(diff_2000_2013_pl_m, bins=20, alpha=0.7, label="Mężczyźni")
ax[0].hist(diff_2000_2013_pl_f, bins=20, alpha=0.7, label="Kobiety")
ax[0].set_title("Polska: różnice w popularności (2000-2013)")
ax[0].set_xlabel("Różnica procentowa")
ax[0].set_ylabel("Liczba imion")
ax[0].legend()

ax[1].hist(diff_2000_2013_usa_m, bins=20, alpha=0.7, label="Mężczyźni")
ax[1].hist(diff_2000_2013_usa_f, bins=20, alpha=0.7, label="Kobiety")
ax[1].set_title("USA: różnice w popularności (2000-2013)")
ax[1].set_xlabel("Różnica procentowa")
ax[1].set_ylabel("Liczba imion")
ax[1].legend()

plt.tight_layout()
plt.show()

print(f'Odpowiedz do zadania 11.2.:')
print(f'W Polsce, procent imion z top 200 jest znacznie wyższy niż w USA, co oznacza, że Polacy nadawali bardziej '
      f'tradycyjne imiona w większym zakresie. Natomiast w USA różnorodność imion jest większa, co widać po spadającym '
      f'udziale imion z top 1000.')
print(f'Zmiany w Polsce mogą wynikać głównie ze zmieniających się zwyczajów i powolnego otwierania się na nowe imiona. '
      f'W USA dodatkowo zauważalny jest wpływ kultury popularnej, trendów medialnych i różnorodności kulturowej.\n')



##### 3 #####
females_pl_df = names_pl_df[names_pl_df['Sex'] == 'F']
males_pl_df = names_pl_df[names_pl_df['Sex'] == 'M']
females_agg = females_pl_df.groupby("Name")["Number"].sum().reset_index()
males_agg = males_pl_df.groupby("Name")["Number"].sum().reset_index()
common_names = pd.merge(females_agg, males_agg, on="Name", suffixes=("_F", "_M"))
common_names["Ratio"] = common_names["Number_F"] / common_names["Number_M"]
neutral_names = common_names[(common_names["Ratio"] >= 0.5) & (common_names["Ratio"] <= 2)]
neutral_names = neutral_names.copy()
neutral_names["Total"] = neutral_names["Number_F"] + neutral_names["Number_M"]
neutral_names = neutral_names.sort_values(by="Total", ascending=False)
print(f'Odpowiedź do zadania 11.3.:')
print(f'2 imiona, które stosunkowo czesto nadawane były dziewczynkom i chłopcom: '
      f'{neutral_names.iloc[0]["Name"]}, '
      f'{neutral_names.iloc[1]["Name"]}')